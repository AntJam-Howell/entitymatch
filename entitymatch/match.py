"""
Embedding generation and similarity computation for entity matching.

Uses sentence-transformers to encode entity names into dense vectors, then
computes cosine similarity to find potential matches.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def load_model(
    model_name: str = "all-MiniLM-L6-v2",
    device: Optional[str] = None,
):
    """Load a sentence-transformer model.

    Parameters
    ----------
    model_name : str
        Name or path of the sentence-transformer model.
        Default is ``"all-MiniLM-L6-v2"`` which provides a good balance
        of speed and quality for entity name matching.
    device : str, optional
        Device to load the model on (``"cpu"``, ``"cuda"``, ``"mps"``).
        If None, auto-detects GPU availability.

    Returns
    -------
    SentenceTransformer
        The loaded model.
    """
    from sentence_transformers import SentenceTransformer

    if device is None:
        import torch

        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    return SentenceTransformer(model_name, device=device)


def encode_names(
    names: list[str] | pd.Series,
    model=None,
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 256,
    show_progress: bool = True,
    device: Optional[str] = None,
) -> dict[str, np.ndarray]:
    """Encode entity names into normalized embedding vectors.

    Deduplicates names before encoding for efficiency. Returns a dictionary
    mapping each unique name to its normalized embedding.

    Parameters
    ----------
    names : list[str] or pd.Series
        Entity names to encode.
    model : SentenceTransformer, optional
        Pre-loaded model. If None, loads ``model_name``.
    model_name : str
        Model to load if ``model`` is None.
    batch_size : int
        Batch size for encoding.
    show_progress : bool
        Whether to show a progress bar.
    device : str, optional
        Device for model loading (only used if ``model`` is None).

    Returns
    -------
    dict[str, np.ndarray]
        Mapping of name string -> normalized embedding vector.
    """
    if model is None:
        model = load_model(model_name, device=device)

    unique_names = list(pd.Series(names).dropna().unique())
    unique_names = [n for n in unique_names if n]  # remove empty strings

    if not unique_names:
        return {}

    embeddings = model.encode(
        unique_names,
        normalize_embeddings=True,
        batch_size=batch_size,
        convert_to_numpy=True,
        show_progress_bar=show_progress,
    )

    return {name: embeddings[i] for i, name in enumerate(unique_names)}


def compute_similarity(
    embeddings_a: dict[str, np.ndarray],
    embeddings_b: dict[str, np.ndarray],
    names_a: Optional[list[str]] = None,
    names_b: Optional[list[str]] = None,
) -> tuple[np.ndarray, list[str], list[str]]:
    """Compute pairwise cosine similarity between two sets of embeddings.

    Parameters
    ----------
    embeddings_a : dict[str, np.ndarray]
        Name -> embedding mapping for set A.
    embeddings_b : dict[str, np.ndarray]
        Name -> embedding mapping for set B.
    names_a : list[str], optional
        Subset of names from A to compare. If None, uses all keys.
    names_b : list[str], optional
        Subset of names from B to compare. If None, uses all keys.

    Returns
    -------
    similarity_matrix : np.ndarray
        Shape ``(len(names_a), len(names_b))`` similarity scores.
    names_a : list[str]
        Ordered list of names for rows.
    names_b : list[str]
        Ordered list of names for columns.
    """
    if names_a is None:
        names_a = list(embeddings_a.keys())
    else:
        names_a = [n for n in names_a if n in embeddings_a]

    if names_b is None:
        names_b = list(embeddings_b.keys())
    else:
        names_b = [n for n in names_b if n in embeddings_b]

    if not names_a or not names_b:
        return np.array([]).reshape(0, 0), names_a, names_b

    ea = np.stack([embeddings_a[n] for n in names_a])
    eb = np.stack([embeddings_b[n] for n in names_b])

    # Dot product of L2-normalized vectors == cosine similarity
    similarity = ea @ eb.T

    return similarity, names_a, names_b
