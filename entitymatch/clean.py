"""
Name cleaning and normalization utilities for entity matching.

Provides functions to standardize company/organization names by removing
legal suffixes, normalizing unicode, stripping punctuation, and preparing
dataframes for blocking and matching.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Optional

import pandas as pd

# Common business entity suffixes to remove during cleaning
BUSINESS_SUFFIXES = [
    " INCORPORATED",
    " CORPORATION",
    " COMPANY",
    " LIMITED",
    " INC",
    " LLC",
    " LLP",
    " LTD",
    " CO.",
    " CO",
    " PLC",
    " CORP",
    " P.C.",
    " LP",
    " L.L.C.",
    " L.L.P.",
    " L.P.",
    " P.A.",
    " N.A.",
    " S.A.",
    " GMBH",
    " AG",
    " PTY",
]

# US state name -> USPS abbreviation mapping
STATE_ABBREVIATIONS = {
    "ALABAMA": "AL", "ALASKA": "AK", "ARIZONA": "AZ", "ARKANSAS": "AR",
    "CALIFORNIA": "CA", "COLORADO": "CO", "CONNECTICUT": "CT", "DELAWARE": "DE",
    "DISTRICT OF COLUMBIA": "DC", "FLORIDA": "FL", "GEORGIA": "GA", "HAWAII": "HI",
    "IDAHO": "ID", "ILLINOIS": "IL", "INDIANA": "IN", "IOWA": "IA",
    "KANSAS": "KS", "KENTUCKY": "KY", "LOUISIANA": "LA", "MAINE": "ME",
    "MARYLAND": "MD", "MASSACHUSETTS": "MA", "MICHIGAN": "MI", "MINNESOTA": "MN",
    "MISSISSIPPI": "MS", "MISSOURI": "MO", "MONTANA": "MT", "NEBRASKA": "NE",
    "NEVADA": "NV", "NEW HAMPSHIRE": "NH", "NEW JERSEY": "NJ", "NEW MEXICO": "NM",
    "NEW YORK": "NY", "NORTH CAROLINA": "NC", "NORTH DAKOTA": "ND", "OHIO": "OH",
    "OKLAHOMA": "OK", "OREGON": "OR", "PENNSYLVANIA": "PA", "RHODE ISLAND": "RI",
    "SOUTH CAROLINA": "SC", "SOUTH DAKOTA": "SD", "TENNESSEE": "TN", "TEXAS": "TX",
    "UTAH": "UT", "VERMONT": "VT", "VIRGINIA": "VA", "WASHINGTON": "WA",
    "WEST VIRGINIA": "WV", "WISCONSIN": "WI", "WYOMING": "WY",
}


def clean_name(name: object) -> str:
    """Clean and normalize an entity name for matching.

    Steps:
    1. Unicode NFKD normalization -> ASCII
    2. Convert to uppercase
    3. Replace punctuation with spaces
    4. Normalize conjunctions (AND, &)
    5. Remove common business suffixes (Inc, LLC, Corp, etc.)
       using word-boundary matching (longest-first)
    6. Collapse whitespace

    Parameters
    ----------
    name : str or object
        The entity name to clean. Handles None/NaN gracefully.

    Returns
    -------
    str
        The cleaned, uppercase name.

    Examples
    --------
    >>> clean_name("McDonald's Corp.")
    'MCDONALD S'
    >>> clean_name("Johnson & Johnson Inc")
    'JOHNSON JOHNSON'
    >>> clean_name("Société Générale S.A.")
    'SOCIETE GENERALE'
    """
    if pd.isna(name):
        return ""
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode("ascii")
    s = s.upper()
    # Strip punctuation first so "Corp." becomes "Corp"
    s = re.sub(r"[^A-Z0-9 ]+", " ", s)
    # Normalize conjunctions
    s = re.sub(r"\b(AND|&)\b", " ", s)
    # Remove business suffixes using word-boundary regex (longest first to
    # avoid partial matches, e.g., "CO" eating part of "CORP")
    sorted_suffixes = sorted(
        (sf.strip() for sf in BUSINESS_SUFFIXES),
        key=len,
        reverse=True,
    )
    suffix_pattern = r"\b(?:" + "|".join(re.escape(sf) for sf in sorted_suffixes) + r")\b"
    s = re.sub(suffix_pattern, "", s)
    return re.sub(r"\s+", " ", s).strip()


def normalize_state(state: str) -> str:
    """Normalize a US state name or abbreviation to a 2-letter USPS code.

    Parameters
    ----------
    state : str
        Full state name or abbreviation (e.g., "California" or "CA").

    Returns
    -------
    str
        Two-letter USPS abbreviation, or the original value uppercased
        if not recognized.

    Examples
    --------
    >>> normalize_state("California")
    'CA'
    >>> normalize_state("ca")
    'CA'
    >>> normalize_state("NY")
    'NY'
    """
    s = str(state).strip().upper()
    if len(s) == 2:
        return s
    return STATE_ABBREVIATIONS.get(s, s)


def prepare_dataframe(
    df: pd.DataFrame,
    name_col: str,
    city_col: Optional[str] = None,
    state_col: Optional[str] = None,
    id_col: Optional[str] = None,
) -> pd.DataFrame:
    """Prepare a dataframe for entity matching.

    Creates standardized columns used by the matching pipeline:
    - ``name_clean``: cleaned entity name
    - ``city``: uppercased city
    - ``state``: USPS-normalized state abbreviation
    - ``blocking_key``: ``CITY|STATE`` geographic blocking key
    - ``entity_id``: unique identifier (uses existing ID column or index)

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe containing entity records.
    name_col : str
        Column name containing entity/company names.
    city_col : str, optional
        Column name containing city information.
    state_col : str, optional
        Column name containing state information.
    id_col : str, optional
        Column name containing unique entity identifiers.
        If None, the dataframe index is used.

    Returns
    -------
    pd.DataFrame
        Copy of the input dataframe with added standardized columns.
    """
    out = df.copy()

    # Clean names
    out["name_clean"] = out[name_col].map(clean_name)

    # Normalize geography
    if city_col and city_col in out.columns:
        out["city"] = out[city_col].fillna("").astype(str).str.upper().str.strip()
    else:
        out["city"] = ""

    if state_col and state_col in out.columns:
        out["state"] = out[state_col].fillna("").astype(str).apply(normalize_state)
    else:
        out["state"] = ""

    # Build blocking key
    out["blocking_key"] = out["city"] + "|" + out["state"]

    # Entity ID
    if id_col and id_col in out.columns:
        out["entity_id"] = out[id_col].astype(str)
    else:
        out["entity_id"] = out.index.astype(str)

    return out
