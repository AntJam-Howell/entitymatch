"""
End-to-end entity matching pipeline.

Combines name cleaning, geographic blocking, embedding-based similarity,
and optional LLM validation into a single high-level interface.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from entitymatch.block import blocked_match, two_tier_blocking_match
from entitymatch.clean import prepare_dataframe
from entitymatch.llm_validate import validate_matches
from entitymatch.match import encode_names, load_model
from entitymatch.utils import apply_acceptance_criteria


def match_entities(
    df_left: pd.DataFrame,
    df_right: pd.DataFrame,
    left_name_col: str,
    right_name_col: str,
    left_id_col: Optional[str] = None,
    right_id_col: Optional[str] = None,
    left_city_col: Optional[str] = None,
    right_city_col: Optional[str] = None,
    left_state_col: Optional[str] = None,
    right_state_col: Optional[str] = None,
    model_name: str = "all-MiniLM-L6-v2",
    top_k: int = 3,
    threshold: float = 0.65,
    auto_accept_threshold: float = 0.85,
    use_llm: bool = False,
    llm_provider: str = "openai",
    llm_model: Optional[str] = None,
    llm_api_key: Optional[str] = None,
    llm_min_score: float = 0.75,
    llm_max_score: float = 0.90,
    llm_batch_size: int = 20,
    batch_size: int = 256,
    show_progress: bool = True,
    device: Optional[str] = None,
) -> pd.DataFrame:
    """Run the full entity matching pipeline.

    This is the main entry point for matching two datasets. It:

    1. Cleans and normalizes entity names
    2. Applies two-tier geographic blocking (city+state, then state fallback)
    3. Computes embedding similarity within each block
    4. Optionally validates gray-zone matches with an LLM
    5. Applies acceptance criteria

    Parameters
    ----------
    df_left, df_right : pd.DataFrame
        The two datasets to match.
    left_name_col, right_name_col : str
        Column names containing entity names.
    left_id_col, right_id_col : str, optional
        Column names for unique entity IDs. If None, uses the index.
    left_city_col, right_city_col : str, optional
        Column names for city. Required for geographic blocking.
    left_state_col, right_state_col : str, optional
        Column names for state. Required for geographic blocking.
    model_name : str
        Sentence-transformer model for embeddings.
    top_k : int
        Number of top matches per entity per block.
    threshold : float
        Minimum similarity score to keep a match candidate.
    auto_accept_threshold : float
        Score at or above which matches are auto-accepted.
    use_llm : bool
        Whether to validate gray-zone matches with an LLM.
    llm_provider : str
        ``"openai"`` or ``"anthropic"``.
    llm_model : str, optional
        LLM model name.
    llm_api_key : str, optional
        LLM API key (or set via environment variable).
    llm_min_score, llm_max_score : float
        Score range for LLM validation.
    llm_batch_size : int
        Concurrent LLM API requests.
    batch_size : int
        Embedding encoding batch size.
    show_progress : bool
        Show progress bars during encoding.
    device : str, optional
        Device for the embedding model.

    Returns
    -------
    pd.DataFrame
        Accepted matches with columns: ``left_id``, ``right_id``,
        ``left_name``, ``right_name``, ``score``, ``block``,
        ``block_level``, ``accept_reason``, and optionally
        ``llm_match`` and ``llm_validated``.
    """
    # Step 1: Prepare dataframes
    left = prepare_dataframe(
        df_left,
        name_col=left_name_col,
        city_col=left_city_col,
        state_col=left_state_col,
        id_col=left_id_col,
    )
    right = prepare_dataframe(
        df_right,
        name_col=right_name_col,
        city_col=right_city_col,
        state_col=right_state_col,
        id_col=right_id_col,
    )

    # Step 2: Encode names
    model = load_model(model_name, device=device)
    emb_left = encode_names(
        left["name_clean"], model=model, batch_size=batch_size,
        show_progress=show_progress,
    )
    emb_right = encode_names(
        right["name_clean"], model=model, batch_size=batch_size,
        show_progress=show_progress,
    )

    # Step 3: Blocking + matching
    has_geo = (left_city_col or left_state_col) and (right_city_col or right_state_col)

    if has_geo:
        matches = two_tier_blocking_match(
            left, right,
            embeddings_left=emb_left,
            embeddings_right=emb_right,
            top_k=top_k,
            threshold=threshold,
        )
    else:
        # No geographic info — match without blocking (full cross-comparison)
        matches = blocked_match(
            left, right,
            embeddings_left=emb_left,
            embeddings_right=emb_right,
            block_col="state",  # will only have empty strings
            top_k=top_k,
            threshold=threshold,
        )
        matches["block_level"] = "none"

    if matches.empty:
        return matches

    # Step 4: Optional LLM validation
    if use_llm:
        matches = validate_matches(
            matches,
            left_name_col="left_name",
            right_name_col="right_name",
            score_col="score",
            min_score=llm_min_score,
            max_score=llm_max_score,
            provider=llm_provider,
            model=llm_model,
            api_key=llm_api_key,
            batch_size=llm_batch_size,
        )

    # Step 5: Apply acceptance criteria
    accepted = apply_acceptance_criteria(
        matches,
        auto_accept_threshold=auto_accept_threshold,
        llm_score_col="score",
        llm_match_col="llm_match",
        llm_min=llm_min_score,
        llm_max=llm_max_score,
    )

    return accepted


class EntityMatcher:
    """Configurable entity matcher with reusable model and settings.

    This class holds the configuration and loaded model, allowing you to
    match multiple dataset pairs without reloading the model each time.

    Parameters
    ----------
    model_name : str
        Sentence-transformer model.
    top_k : int
        Top matches per entity per block.
    threshold : float
        Minimum similarity score.
    auto_accept_threshold : float
        Score for auto-acceptance.
    use_llm : bool
        Enable LLM validation.
    llm_provider : str
        ``"openai"`` or ``"anthropic"``.
    llm_model : str, optional
        LLM model name.
    llm_api_key : str, optional
        LLM API key.
    llm_min_score, llm_max_score : float
        LLM validation score range.
    llm_batch_size : int
        Concurrent LLM requests.
    batch_size : int
        Embedding batch size.
    device : str, optional
        Model device.

    Examples
    --------
    >>> matcher = EntityMatcher(use_llm=False)
    >>> results = matcher.match(
    ...     df_a, df_b,
    ...     left_name_col="company_name",
    ...     right_name_col="firm_name",
    ...     left_city_col="city",
    ...     right_city_col="city",
    ...     left_state_col="state",
    ...     right_state_col="state",
    ... )
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        top_k: int = 3,
        threshold: float = 0.65,
        auto_accept_threshold: float = 0.85,
        use_llm: bool = False,
        llm_provider: str = "openai",
        llm_model: Optional[str] = None,
        llm_api_key: Optional[str] = None,
        llm_min_score: float = 0.75,
        llm_max_score: float = 0.90,
        llm_batch_size: int = 20,
        batch_size: int = 256,
        device: Optional[str] = None,
    ):
        self.model_name = model_name
        self.top_k = top_k
        self.threshold = threshold
        self.auto_accept_threshold = auto_accept_threshold
        self.use_llm = use_llm
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self.llm_api_key = llm_api_key
        self.llm_min_score = llm_min_score
        self.llm_max_score = llm_max_score
        self.llm_batch_size = llm_batch_size
        self.batch_size = batch_size
        self.device = device
        self._model = None

    @property
    def model(self):
        """Lazy-load the sentence transformer model."""
        if self._model is None:
            self._model = load_model(self.model_name, self.device)
        return self._model

    def match(
        self,
        df_left: pd.DataFrame,
        df_right: pd.DataFrame,
        left_name_col: str,
        right_name_col: str,
        left_id_col: Optional[str] = None,
        right_id_col: Optional[str] = None,
        left_city_col: Optional[str] = None,
        right_city_col: Optional[str] = None,
        left_state_col: Optional[str] = None,
        right_state_col: Optional[str] = None,
        show_progress: bool = True,
    ) -> pd.DataFrame:
        """Match two entity datasets.

        See :func:`match_entities` for full parameter documentation.
        """
        return match_entities(
            df_left=df_left,
            df_right=df_right,
            left_name_col=left_name_col,
            right_name_col=right_name_col,
            left_id_col=left_id_col,
            right_id_col=right_id_col,
            left_city_col=left_city_col,
            right_city_col=right_city_col,
            left_state_col=left_state_col,
            right_state_col=right_state_col,
            model_name=self.model_name,
            top_k=self.top_k,
            threshold=self.threshold,
            auto_accept_threshold=self.auto_accept_threshold,
            use_llm=self.use_llm,
            llm_provider=self.llm_provider,
            llm_model=self.llm_model,
            llm_api_key=self.llm_api_key,
            llm_min_score=self.llm_min_score,
            llm_max_score=self.llm_max_score,
            llm_batch_size=self.llm_batch_size,
            batch_size=self.batch_size,
            show_progress=show_progress,
            device=self.device,
        )
