"""
Utility functions for entitymatch.
"""

from __future__ import annotations

import pandas as pd


def find_name_column(df: pd.DataFrame) -> str:
    """Auto-detect the entity name column in a dataframe.

    Checks for common column names used for company/entity names.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.

    Returns
    -------
    str
        Detected column name.

    Raises
    ------
    KeyError
        If no recognized name column is found.
    """
    candidates = [
        "company_name",
        "company",
        "name",
        "entity_name",
        "business_name",
        "firm_name",
        "organization",
        "org_name",
        "employer_name",
        "EMPLOYER_NAME",
        "company_raw",
    ]
    for c in candidates:
        if c in df.columns:
            return c
    # Case-insensitive fallback
    lower_cols = {col.lower(): col for col in df.columns}
    for c in candidates:
        if c.lower() in lower_cols:
            return lower_cols[c.lower()]
    raise KeyError(
        f"No entity name column found. Expected one of: {candidates}. "
        f"Found columns: {list(df.columns)}"
    )


def apply_acceptance_criteria(
    matches_df: pd.DataFrame,
    auto_accept_threshold: float = 0.85,
    llm_score_col: str = "score",
    llm_match_col: str = "llm_match",
    llm_min: float = 0.75,
    llm_max: float = 0.90,
) -> pd.DataFrame:
    """Apply final acceptance criteria to a matches dataframe.

    A match is accepted if:
    1. Its similarity score >= ``auto_accept_threshold``, OR
    2. Its score is in ``[llm_min, llm_max)`` AND the LLM confirmed it.

    Parameters
    ----------
    matches_df : pd.DataFrame
        Matches with score and optional LLM validation columns.
    auto_accept_threshold : float
        Score at or above which matches are auto-accepted.
    llm_score_col : str
        Column with similarity scores.
    llm_match_col : str
        Column with LLM validation boolean (True/False).
    llm_min, llm_max : float
        Score range for LLM-validated matches.

    Returns
    -------
    pd.DataFrame
        Filtered dataframe containing only accepted matches, with an
        ``accept_reason`` column.
    """
    df = matches_df.copy()

    high_sim = df[llm_score_col] >= auto_accept_threshold

    has_llm = llm_match_col in df.columns
    if has_llm:
        llm_confirmed = (
            (df[llm_score_col] >= llm_min)
            & (df[llm_score_col] < llm_max)
            & (df[llm_match_col] == True)  # noqa: E712
        )
    else:
        llm_confirmed = pd.Series(False, index=df.index)

    accepted = df[high_sim | llm_confirmed].copy()
    accepted["accept_reason"] = "similarity"
    accepted.loc[llm_confirmed[high_sim | llm_confirmed], "accept_reason"] = "llm_validated"
    # Matches that meet BOTH criteria get "similarity"
    accepted.loc[high_sim[high_sim | llm_confirmed], "accept_reason"] = "similarity"

    return accepted.reset_index(drop=True)
