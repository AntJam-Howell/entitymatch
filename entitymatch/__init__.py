"""
entitymatch - Semantic entity matching with geographic blocking and LLM validation.

A Python package for matching entity records across two datasets using
sentence-transformer embeddings, geographic blocking, cosine similarity scoring,
and optional LLM-based validation for gray-zone matches.
"""

from entitymatch.clean import clean_name, normalize_state, prepare_dataframe
from entitymatch.block import build_blocking_key, blocked_match
from entitymatch.match import load_model, encode_names, compute_similarity
from entitymatch.llm_validate import validate_matches, validate_pair
from entitymatch.pipeline import match_entities, EntityMatcher

__version__ = "0.1.1"

__all__ = [
    "clean_name",
    "normalize_state",
    "prepare_dataframe",
    "build_blocking_key",
    "blocked_match",
    "load_model",
    "encode_names",
    "compute_similarity",
    "validate_matches",
    "validate_pair",
    "match_entities",
    "EntityMatcher",
]
