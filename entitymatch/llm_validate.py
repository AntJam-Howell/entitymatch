"""
LLM-based validation for gray-zone entity matches.

Sends match pairs to an LLM (OpenAI or Anthropic) to get a second opinion
on whether two entity names refer to the same real-world entity. This is
especially useful for matches in the "gray zone" (e.g., similarity 0.75-0.90)
where embedding similarity alone is ambiguous.
"""

from __future__ import annotations

import asyncio
import os
import time
from typing import Optional

import pandas as pd


def _build_prompt(
    name_a: str,
    name_b: str,
    similarity: float,
    city_a: str = "",
    state_a: str = "",
    city_b: str = "",
    state_b: str = "",
) -> str:
    """Build the LLM prompt for match validation."""
    parts = [
        "You are checking if two business records refer to the same company.",
        "Consider spelling variations, abbreviations, and whether they could be the same company.",
        "Reply with exactly one word: MATCH or NO_MATCH.",
        "",
        f"Company A: {name_a}",
    ]
    if city_a or state_a:
        parts.append(f"  Location: {city_a}, {state_a}")
    parts.append(f"Company B: {name_b}")
    if city_b or state_b:
        parts.append(f"  Location: {city_b}, {state_b}")
    parts.append(f"Embedding similarity score: {similarity:.3f}")
    return "\n".join(parts)


def _parse_response(text: str) -> bool:
    """Parse LLM response to determine if it's a match."""
    text = text.strip().upper()
    if "NO_MATCH" in text or "NO MATCH" in text:
        return False
    if "MATCH" in text:
        return True
    return False


async def _call_openai(
    session,
    prompt: str,
    api_key: str,
    model: str = "gpt-4o-mini",
    max_retries: int = 3,
) -> bool:
    """Call OpenAI API for match validation."""
    import aiohttp

    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    body = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are a data matching assistant. Reply with only MATCH or NO_MATCH.",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
    }

    for attempt in range(max_retries):
        try:
            async with session.post(url, headers=headers, json=body, timeout=30) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    text = (
                        data.get("choices", [{}])[0]
                        .get("message", {})
                        .get("content", "")
                    )
                    return _parse_response(text)
                if resp.status == 429 or resp.status >= 500:
                    await asyncio.sleep(2**attempt)
                    continue
                return False
        except Exception:
            if attempt < max_retries - 1:
                await asyncio.sleep(2**attempt)
            continue
    return False


async def _call_anthropic(
    session,
    prompt: str,
    api_key: str,
    model: str = "claude-haiku-4-5-20251001",
    max_retries: int = 3,
) -> bool:
    """Call Anthropic API for match validation."""
    url = "https://api.anthropic.com/v1/messages"
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json",
    }
    body = {
        "model": model,
        "max_tokens": 10,
        "system": "You are a data matching assistant. Reply with only MATCH or NO_MATCH.",
        "messages": [{"role": "user", "content": prompt}],
    }

    for attempt in range(max_retries):
        try:
            async with session.post(url, headers=headers, json=body, timeout=30) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    text = data.get("content", [{}])[0].get("text", "")
                    return _parse_response(text)
                if resp.status == 429 or resp.status >= 500:
                    await asyncio.sleep(2**attempt)
                    continue
                return False
        except Exception:
            if attempt < max_retries - 1:
                await asyncio.sleep(2**attempt)
            continue
    return False


def validate_pair(
    name_a: str,
    name_b: str,
    similarity: float,
    provider: str = "openai",
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    city_a: str = "",
    state_a: str = "",
    city_b: str = "",
    state_b: str = "",
) -> bool:
    """Validate a single entity match pair using an LLM (synchronous).

    Parameters
    ----------
    name_a, name_b : str
        Entity names to compare.
    similarity : float
        Pre-computed embedding similarity score.
    provider : str
        LLM provider: ``"openai"`` or ``"anthropic"``.
    model : str, optional
        Model name. Defaults to ``"gpt-4o-mini"`` (OpenAI) or
        ``"claude-haiku-4-5-20251001"`` (Anthropic).
    api_key : str, optional
        API key. If None, reads from ``OPENAI_API_KEY`` or
        ``ANTHROPIC_API_KEY`` environment variable.
    city_a, state_a, city_b, state_b : str
        Optional location info for context.

    Returns
    -------
    bool
        True if the LLM confirms the match.
    """
    import requests

    if api_key is None:
        env_var = "OPENAI_API_KEY" if provider == "openai" else "ANTHROPIC_API_KEY"
        api_key = os.environ.get(env_var)
        if not api_key:
            raise ValueError(
                f"No API key provided and {env_var} not set in environment."
            )

    prompt = _build_prompt(
        name_a, name_b, similarity, city_a, state_a, city_b, state_b
    )

    if provider == "openai":
        if model is None:
            model = "gpt-4o-mini"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        body = {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a data matching assistant. Reply with only MATCH or NO_MATCH.",
                },
                {"role": "user", "content": prompt},
            ],
            "temperature": 0,
        }
        resp = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=body,
            timeout=60,
        )
        resp.raise_for_status()
        text = (
            resp.json()
            .get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )
        return _parse_response(text)

    elif provider == "anthropic":
        if model is None:
            model = "claude-haiku-4-5-20251001"
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
        body = {
            "model": model,
            "max_tokens": 10,
            "system": "You are a data matching assistant. Reply with only MATCH or NO_MATCH.",
            "messages": [{"role": "user", "content": prompt}],
        }
        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers=headers,
            json=body,
            timeout=60,
        )
        resp.raise_for_status()
        text = resp.json().get("content", [{}])[0].get("text", "")
        return _parse_response(text)

    else:
        raise ValueError(f"Unsupported provider: {provider!r}. Use 'openai' or 'anthropic'.")


async def _validate_batch_async(
    matches: list[dict],
    provider: str,
    api_key: str,
    model: str,
    batch_size: int,
    max_retries: int,
) -> list[bool]:
    """Validate a batch of matches asynchronously."""
    import aiohttp

    call_fn = _call_openai if provider == "openai" else _call_anthropic

    connector = aiohttp.TCPConnector(limit=batch_size)
    timeout = aiohttp.ClientTimeout(total=60)
    results = [False] * len(matches)

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        for start in range(0, len(matches), batch_size):
            batch = matches[start : start + batch_size]
            tasks = []
            for m in batch:
                prompt = _build_prompt(
                    m["name_a"],
                    m["name_b"],
                    m["similarity"],
                    m.get("city_a", ""),
                    m.get("state_a", ""),
                    m.get("city_b", ""),
                    m.get("state_b", ""),
                )
                tasks.append(call_fn(session, prompt, api_key, model, max_retries))

            batch_results = await asyncio.gather(*tasks, return_exceptions=True)
            for i, result in enumerate(batch_results):
                idx = start + i
                results[idx] = result if isinstance(result, bool) else False

    return results


def validate_matches(
    matches_df: pd.DataFrame,
    left_name_col: str = "left_name",
    right_name_col: str = "right_name",
    score_col: str = "score",
    min_score: float = 0.75,
    max_score: float = 0.90,
    provider: str = "openai",
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    batch_size: int = 20,
    max_retries: int = 3,
) -> pd.DataFrame:
    """Validate gray-zone matches using LLM calls.

    Filters the matches dataframe to the score range ``[min_score, max_score)``
    and sends each pair to an LLM for validation. Results are added as a
    ``llm_match`` boolean column.

    Parameters
    ----------
    matches_df : pd.DataFrame
        Dataframe of match candidates with name and score columns.
    left_name_col, right_name_col : str
        Column names for entity names.
    score_col : str
        Column name for similarity score.
    min_score, max_score : float
        Score range for LLM validation. Matches outside this range
        are not sent to the LLM.
    provider : str
        ``"openai"`` or ``"anthropic"``.
    model : str, optional
        Model name override.
    api_key : str, optional
        API key override. If None, reads from environment.
    batch_size : int
        Concurrent API requests.
    max_retries : int
        Retries per API call.

    Returns
    -------
    pd.DataFrame
        Copy of ``matches_df`` with added columns:

        - ``llm_match``: True if LLM confirmed match (only for gray-zone rows)
        - ``llm_validated``: True if the row was sent to LLM
    """
    if api_key is None:
        env_var = "OPENAI_API_KEY" if provider == "openai" else "ANTHROPIC_API_KEY"
        api_key = os.environ.get(env_var)
        if not api_key:
            raise ValueError(
                f"No API key provided and {env_var} not set in environment."
            )

    if model is None:
        model = "gpt-4o-mini" if provider == "openai" else "claude-haiku-4-5-20251001"

    result = matches_df.copy()
    result["llm_match"] = False
    result["llm_validated"] = False

    # Filter to gray zone
    gray_mask = (result[score_col] >= min_score) & (result[score_col] < max_score)
    gray_zone = result[gray_mask]

    if gray_zone.empty:
        return result

    # Build match list
    match_list = []
    for _, row in gray_zone.iterrows():
        m = {
            "name_a": str(row[left_name_col]),
            "name_b": str(row[right_name_col]),
            "similarity": float(row[score_col]),
        }
        # Include location if available
        for col, key in [
            ("city_left", "city_a"),
            ("state_left", "state_a"),
            ("city_right", "city_b"),
            ("state_right", "state_b"),
        ]:
            if col in row.index:
                m[key] = str(row[col])
        match_list.append(m)

    # Run async validation
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # Inside Jupyter or another async context
        import nest_asyncio

        nest_asyncio.apply()

    llm_results = asyncio.run(
        _validate_batch_async(
            match_list, provider, api_key, model, batch_size, max_retries
        )
    )

    # Apply results
    result.loc[gray_mask, "llm_match"] = llm_results
    result.loc[gray_mask, "llm_validated"] = True

    return result
