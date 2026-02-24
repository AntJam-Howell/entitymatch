"""Tests for entitymatch.clean module."""

import pandas as pd
import pytest

from entitymatch.clean import clean_name, normalize_state, prepare_dataframe


class TestCleanName:
    def test_basic_cleaning(self):
        # Apostrophe becomes a space: "McDonald's" -> "MCDONALD S"
        assert clean_name("McDonald's Corp.") == "MCDONALD S"

    def test_removes_inc(self):
        assert clean_name("Apple Inc") == "APPLE"

    def test_removes_llc(self):
        assert clean_name("Acme LLC") == "ACME"

    def test_removes_corporation(self):
        assert clean_name("Microsoft Corporation") == "MICROSOFT"

    def test_removes_company(self):
        assert clean_name("Ford Motor Company") == "FORD MOTOR"

    def test_removes_multiple_suffixes(self):
        assert clean_name("Acme Corp Inc") == "ACME"

    def test_normalizes_ampersand(self):
        assert clean_name("Johnson & Johnson") == "JOHNSON JOHNSON"

    def test_normalizes_and(self):
        assert clean_name("Johnson AND Johnson") == "JOHNSON JOHNSON"

    def test_unicode_normalization(self):
        assert clean_name("Societe Generale") == "SOCIETE GENERALE"

    def test_handles_none(self):
        assert clean_name(None) == ""

    def test_handles_nan(self):
        assert clean_name(float("nan")) == ""

    def test_handles_empty_string(self):
        assert clean_name("") == ""

    def test_collapses_whitespace(self):
        assert clean_name("  ACME   CORP  ") == "ACME"

    def test_strips_punctuation(self):
        assert clean_name("A.B.C. Industries") == "A B C INDUSTRIES"

    def test_preserves_numbers(self):
        assert clean_name("3M Company") == "3M"


class TestNormalizeState:
    def test_full_name(self):
        assert normalize_state("California") == "CA"

    def test_abbreviation_passthrough(self):
        assert normalize_state("CA") == "CA"

    def test_lowercase_abbreviation(self):
        assert normalize_state("ca") == "CA"

    def test_full_name_uppercase(self):
        assert normalize_state("NEW YORK") == "NY"

    def test_unknown_returns_uppercased(self):
        assert normalize_state("Ontario") == "ONTARIO"

    def test_whitespace_stripped(self):
        assert normalize_state("  TX  ") == "TX"


class TestPrepareDataframe:
    @pytest.fixture
    def sample_df(self):
        return pd.DataFrame(
            {
                "company_name": ["Apple Inc", "Google LLC", "Amazon.com Inc"],
                "city": ["Cupertino", "Mountain View", "Seattle"],
                "state": ["California", "CA", "WA"],
                "duns": ["001", "002", "003"],
            }
        )

    def test_adds_name_clean(self, sample_df):
        result = prepare_dataframe(sample_df, name_col="company_name")
        assert "name_clean" in result.columns
        assert result["name_clean"].iloc[0] == "APPLE"

    def test_adds_blocking_key(self, sample_df):
        result = prepare_dataframe(
            sample_df,
            name_col="company_name",
            city_col="city",
            state_col="state",
        )
        assert "blocking_key" in result.columns
        assert result["blocking_key"].iloc[0] == "CUPERTINO|CA"

    def test_normalizes_state(self, sample_df):
        result = prepare_dataframe(
            sample_df,
            name_col="company_name",
            state_col="state",
        )
        assert result["state"].iloc[0] == "CA"  # California -> CA

    def test_uses_id_col(self, sample_df):
        result = prepare_dataframe(
            sample_df,
            name_col="company_name",
            id_col="duns",
        )
        assert result["entity_id"].iloc[0] == "001"

    def test_uses_index_if_no_id_col(self, sample_df):
        result = prepare_dataframe(sample_df, name_col="company_name")
        assert result["entity_id"].iloc[0] == "0"

    def test_preserves_original_columns(self, sample_df):
        result = prepare_dataframe(sample_df, name_col="company_name")
        assert "company_name" in result.columns
        assert "city" in result.columns

    def test_handles_missing_city(self, sample_df):
        result = prepare_dataframe(sample_df, name_col="company_name")
        assert (result["city"] == "").all()

    def test_does_not_modify_original(self, sample_df):
        original_cols = list(sample_df.columns)
        prepare_dataframe(sample_df, name_col="company_name")
        assert list(sample_df.columns) == original_cols
