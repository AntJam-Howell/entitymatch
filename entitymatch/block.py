"""
Geographic blocking for entity matching.

Blocking reduces the search space by only comparing entities that share
a geographic attribute (city+state or state). This dramatically improves
both computational efficiency and match precision.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def build_blocking_key(
    df: pd.DataFrame,
    city_col: str = "city",
    state_col: str = "state",
) -> pd.Series:
    """Build a ``CITY|STATE`` blocking key from city and state columns.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with city and state columns.
    city_col : str
        Column name for city.
    state_col : str
        Column name for state.

    Returns
    -------
    pd.Series
        Series of ``"CITY|STATE"`` strings.
    """
    city = df[city_col].fillna("").astype(str).str.upper().str.strip()
    state = df[state_col].fillna("").astype(str).str.upper().str.strip()
    return city + "|" + state


def blocked_match(
    df_left: pd.DataFrame,
    df_right: pd.DataFrame,
    embeddings_left: dict[str, np.ndarray],
    embeddings_right: dict[str, np.ndarray],
    block_col: str = "blocking_key",
    top_k: int = 3,
    threshold: float = 0.65,
    left_name_col: str = "name_clean",
    right_name_col: str = "name_clean",
    left_id_col: str = "entity_id",
    right_id_col: str = "entity_id",
) -> pd.DataFrame:
    """Match entities within geographic blocks using embedding similarity.

    For each block (e.g., city+state), computes cosine similarity between
    all left-side and right-side entity name embeddings, then returns the
    top-K matches above the threshold for each left-side entity.

    Parameters
    ----------
    df_left : pd.DataFrame
        Left dataframe (e.g., dataset A). Must contain ``block_col``,
        ``left_name_col``, and ``left_id_col``.
    df_right : pd.DataFrame
        Right dataframe (e.g., dataset B). Must contain ``block_col``,
        ``right_name_col``, and ``right_id_col``.
    embeddings_left : dict[str, np.ndarray]
        Mapping of cleaned name -> normalized embedding vector for left side.
    embeddings_right : dict[str, np.ndarray]
        Mapping of cleaned name -> normalized embedding vector for right side.
    block_col : str
        Column used for blocking (default: ``"blocking_key"``).
    top_k : int
        Number of top matches to return per left entity per block.
    threshold : float
        Minimum similarity score to include a match.
    left_name_col : str
        Column in ``df_left`` with cleaned names.
    right_name_col : str
        Column in ``df_right`` with cleaned names.
    left_id_col : str
        Column in ``df_left`` with entity IDs.
    right_id_col : str
        Column in ``df_right`` with entity IDs.

    Returns
    -------
    pd.DataFrame
        Match results with columns: ``left_id``, ``right_id``,
        ``left_name``, ``right_name``, ``score``, ``block``.
    """
    # Find shared blocks
    left_blocks = set(df_left[block_col].dropna().unique())
    right_blocks = set(df_right[block_col].dropna().unique())
    shared_blocks = sorted(left_blocks & right_blocks)

    parts: list[pd.DataFrame] = []

    for blk in shared_blocks:
        left_blk = df_left[df_left[block_col] == blk].drop_duplicates(
            subset=[left_id_col, left_name_col]
        )
        right_blk = df_right[df_right[block_col] == blk].drop_duplicates(
            subset=[right_id_col, right_name_col]
        )

        if left_blk.empty or right_blk.empty:
            continue

        # Filter to names that have embeddings
        left_names = [
            n for n in left_blk[left_name_col].tolist() if n in embeddings_left
        ]
        right_names = [
            n for n in right_blk[right_name_col].tolist() if n in embeddings_right
        ]

        if not left_names or not right_names:
            continue

        # Stack embeddings and compute similarity (dot product of normalized vecs)
        ea = np.stack([embeddings_left[n] for n in left_names])
        eb = np.stack([embeddings_right[n] for n in right_names])
        S = ea @ eb.T  # cosine similarity for normalized embeddings

        # Top-K selection
        k = min(top_k, S.shape[1])
        top_indices = S.argsort(axis=1)[:, -k:]

        rows = []
        for i, js in enumerate(top_indices):
            for j in js:
                score = float(S[i, j])
                if score >= threshold:
                    rows.append((left_names[i], blk, score, right_names[j]))

        if not rows:
            continue

        # Use distinct internal column names for left/right names to avoid
        # collisions when left_name_col and right_name_col are identical.
        _lname = "__left_name"
        _rname = "__right_name"

        cand = pd.DataFrame(
            rows, columns=[_lname, block_col, "score", _rname]
        )

        # Join back IDs from left side
        left_info = left_blk[[left_id_col, left_name_col, block_col]].drop_duplicates(
            subset=[left_name_col, block_col]
        )
        cand = cand.merge(
            left_info.rename(columns={left_name_col: _lname}),
            on=[_lname, block_col], how="left",
        )

        # Join back IDs from right side
        right_info = right_blk[
            [right_id_col, right_name_col, block_col]
        ].drop_duplicates(subset=[right_name_col, block_col])
        cand = cand.merge(
            right_info.rename(columns={right_name_col: _rname}),
            on=[_rname, block_col], how="left",
            suffixes=("_left", "_right"),
        )

        # Resolve ID column names after merge (suffixed when both sides
        # share the same id column name, e.g. "entity_id").
        lid = left_id_col if left_id_col in cand.columns else f"{left_id_col}_left"
        rid = right_id_col if right_id_col in cand.columns else f"{right_id_col}_right"

        part = pd.DataFrame({
            "left_id": cand[lid],
            "right_id": cand[rid],
            "left_name": cand[_lname],
            "right_name": cand[_rname],
            "score": cand["score"],
            "block": blk,
        })
        parts.append(part)

    if parts:
        result = pd.concat(parts, ignore_index=True)
        result = result.dropna(subset=["left_id", "right_id"])
        result = result.drop_duplicates(subset=["left_id", "right_id"])
        return result

    return pd.DataFrame(
        columns=["left_id", "right_id", "left_name", "right_name", "score", "block"]
    )


def two_tier_blocking_match(
    df_left: pd.DataFrame,
    df_right: pd.DataFrame,
    embeddings_left: dict[str, np.ndarray],
    embeddings_right: dict[str, np.ndarray],
    top_k: int = 3,
    threshold: float = 0.65,
    left_name_col: str = "name_clean",
    right_name_col: str = "name_clean",
    left_id_col: str = "entity_id",
    right_id_col: str = "entity_id",
    min_city_matches: int = 3,
) -> pd.DataFrame:
    """Two-tier blocking: city+state first, then state-level fallback.

    First matches using city+state blocking. For left-side entities that
    receive fewer than ``min_city_matches`` results, falls back to
    state-level blocking.

    Parameters
    ----------
    df_left : pd.DataFrame
        Left dataframe with ``blocking_key`` and ``state`` columns.
    df_right : pd.DataFrame
        Right dataframe with ``blocking_key`` and ``state`` columns.
    embeddings_left : dict
        Name -> embedding mapping for left side.
    embeddings_right : dict
        Name -> embedding mapping for right side.
    top_k : int
        Top matches per entity per block level.
    threshold : float
        Minimum similarity score.
    left_name_col, right_name_col, left_id_col, right_id_col : str
        Column names (see ``blocked_match``).
    min_city_matches : int
        If a left entity gets fewer than this many city-level matches,
        also run state-level blocking for that entity.

    Returns
    -------
    pd.DataFrame
        Combined match results with an added ``block_level`` column
        (``"citystate"`` or ``"state"``).
    """
    common_kwargs = dict(
        embeddings_left=embeddings_left,
        embeddings_right=embeddings_right,
        top_k=top_k,
        threshold=threshold,
        left_name_col=left_name_col,
        right_name_col=right_name_col,
        left_id_col=left_id_col,
        right_id_col=right_id_col,
    )

    # City+state blocking
    city_matches = blocked_match(
        df_left, df_right, block_col="blocking_key", **common_kwargs
    )
    city_matches["block_level"] = "citystate"

    # Identify left IDs that need state-level fallback
    matched_left_ids = set(city_matches["left_id"].unique())
    all_left_ids = set(df_left[left_id_col].unique())
    unmatched = all_left_ids - matched_left_ids

    # Also include those with fewer than min_city_matches
    if not city_matches.empty:
        match_counts = city_matches.groupby("left_id").size()
        sparse = set(match_counts[match_counts < min_city_matches].index)
        needs_state = unmatched | sparse
    else:
        needs_state = all_left_ids

    if needs_state:
        state_left = df_left[df_left[left_id_col].isin(needs_state)]
        state_matches = blocked_match(
            state_left, df_right, block_col="state", **common_kwargs
        )
        state_matches["block_level"] = "state"

        # Exclude any left_id/right_id pairs already found at city level
        if not city_matches.empty and not state_matches.empty:
            existing = set(
                zip(city_matches["left_id"], city_matches["right_id"])
            )
            mask = [
                (lid, rid) not in existing
                for lid, rid in zip(
                    state_matches["left_id"], state_matches["right_id"]
                )
            ]
            state_matches = state_matches[mask]

        combined = pd.concat([city_matches, state_matches], ignore_index=True)
    else:
        combined = city_matches

    return combined.drop_duplicates(subset=["left_id", "right_id"]).reset_index(
        drop=True
    )
