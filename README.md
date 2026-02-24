# entitymatch

A Python package for matching entity records across two datasets using **semantic embeddings**, **geographic blocking**, and optional **LLM-based validation**.

Designed for researchers and data engineers who need to link records across messy, real-world datasets — company names with abbreviations, spelling variations, and inconsistent formatting.

## How It Works

The matching pipeline has four stages:

```
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  1. Name Cleaning │ --> │  2. Geographic    │ --> │  3. Embedding    │ --> │  4. LLM           │
│                  │     │     Blocking      │     │     Similarity   │     │     Validation    │
│  Normalize names │     │  Restrict matches │     │  Rank candidates │     │  Confirm gray     │
│  Remove suffixes │     │  to same area     │     │  by similarity   │     │  zone matches     │
└──────────────────┘     └──────────────────┘     └──────────────────┘     └──────────────────┘
```

### Stage 1: Name Cleaning

Entity names are normalized to improve match quality:

- Unicode normalization (accented characters → ASCII)
- Uppercase conversion
- Removal of business suffixes (Inc, LLC, Corp, Ltd, etc.)
- Punctuation stripping
- Conjunction normalization (AND, &)

### Stage 2: Geographic Blocking

Blocking restricts comparisons to entities in the same geographic area, which:

- **Reduces false positives** — "Acme Corp" in Texas isn't the same as "Acme Corp" in New York
- **Speeds up matching** — comparing within blocks is O(n×m) per block instead of O(N×M) total

Two-tier approach:
1. **City+State** blocking (primary): exact city and state match
2. **State-level** fallback: for entities with insufficient city-level matches

### Stage 3: Embedding Similarity

Uses [sentence-transformers](https://www.sbert.net/) to encode entity names into dense vectors, then ranks matches by cosine similarity:

- Default model: `all-MiniLM-L6-v2` (fast, good quality)
- Captures semantic similarity: "IBM" ≈ "International Business Machines"
- Handles abbreviations, spelling variations, and word reordering
- Returns top-K candidates per entity above a configurable threshold

### Stage 4: LLM Validation (Optional)

For matches in the "gray zone" (e.g., similarity 0.75–0.90), an LLM provides a second opinion:

- Sends name pairs to an LLM with the prompt: "Do these refer to the same company?"
- Supports **OpenAI** and **Anthropic** APIs
- Async batching for throughput (20 concurrent requests by default)
- Typical cost: ~$0.50 per 20,000 validations with `gpt-4o-mini`

### Final Acceptance

A match is accepted if:
1. **Similarity ≥ 0.85** (auto-accept), OR
2. **0.75 ≤ similarity < 0.90** AND **LLM confirms match**

All thresholds are configurable.

## Installation

```bash
pip install entitymatch
```

Or install from source:

```bash
git clone https://github.com/AntJam-Howell/entitymatch.git
cd entitymatch
pip install -e ".[all]"
```

### Optional dependencies

```bash
# For LLM validation (OpenAI or Anthropic)
pip install entitymatch[llm]

# For development
pip install entitymatch[dev]
```

## Quick Start

```python
import pandas as pd
from entitymatch import match_entities

# Two datasets with company names and locations
companies_a = pd.DataFrame({
    "id": ["A1", "A2", "A3"],
    "company_name": ["McDonald's Corporation", "IBM", "Walmart Inc"],
    "city": ["Chicago", "Armonk", "Bentonville"],
    "state": ["IL", "NY", "AR"],
})

companies_b = pd.DataFrame({
    "id": ["B1", "B2", "B3", "B4"],
    "company_name": ["McDonalds Corp", "International Business Machines", "Wal-Mart Stores", "Target Corp"],
    "city": ["Chicago", "Armonk", "Bentonville", "Minneapolis"],
    "state": ["IL", "NY", "AR", "MN"],
})

# Run matching (without LLM validation)
results = match_entities(
    df_left=companies_a,
    df_right=companies_b,
    left_name_col="company_name",
    right_name_col="company_name",
    left_id_col="id",
    right_id_col="id",
    left_city_col="city",
    right_city_col="city",
    left_state_col="state",
    right_state_col="state",
)

print(results[["left_id", "right_id", "left_name", "right_name", "score"]])
```

### With LLM Validation

```python
import os

# Set your API key
os.environ["OPENAI_API_KEY"] = "your-key-here"

results = match_entities(
    df_left=companies_a,
    df_right=companies_b,
    left_name_col="company_name",
    right_name_col="company_name",
    left_id_col="id",
    right_id_col="id",
    left_city_col="city",
    right_city_col="city",
    left_state_col="state",
    right_state_col="state",
    use_llm=True,
    llm_provider="openai",       # or "anthropic"
    llm_model="gpt-4o-mini",     # or "claude-haiku-4-5-20251001"
)
```

### Using the EntityMatcher Class

For matching multiple dataset pairs without reloading the model:

```python
from entitymatch import EntityMatcher

matcher = EntityMatcher(
    model_name="all-MiniLM-L6-v2",
    top_k=3,
    threshold=0.65,
    auto_accept_threshold=0.85,
)

# Match dataset pairs
results_2023 = matcher.match(df_left, df_right_2023, left_name_col="name", right_name_col="name")
results_2024 = matcher.match(df_left, df_right_2024, left_name_col="name", right_name_col="name")
```

## Using Individual Components

Each stage of the pipeline is available as a standalone module:

```python
from entitymatch.clean import clean_name, prepare_dataframe
from entitymatch.match import load_model, encode_names
from entitymatch.block import blocked_match, two_tier_blocking_match
from entitymatch.llm_validate import validate_matches, validate_pair
from entitymatch.utils import apply_acceptance_criteria

# Clean a single name
clean_name("McDonald's Corp.")  # → "MCDONALD S"

# Prepare a dataframe
df = prepare_dataframe(raw_df, name_col="company", city_col="city", state_col="state")

# Encode names
model = load_model()
embeddings = encode_names(df["name_clean"], model=model)

# Validate a single pair with LLM
is_match = validate_pair("McDonalds", "McDonald's Corporation", similarity=0.82, provider="openai")
```

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_name` | `"all-MiniLM-L6-v2"` | Sentence-transformer model |
| `top_k` | `3` | Top matches per entity per block |
| `threshold` | `0.65` | Minimum similarity to keep a candidate |
| `auto_accept_threshold` | `0.85` | Score for automatic acceptance |
| `llm_min_score` | `0.75` | Lower bound of LLM validation range |
| `llm_max_score` | `0.90` | Upper bound of LLM validation range |
| `llm_batch_size` | `20` | Concurrent LLM API calls |

### Threshold Strategy

| Similarity | Treatment | Rationale |
|-----------|-----------|-----------|
| **≥ 0.90** | Auto-accept | Very high confidence |
| **0.85–0.90** | Auto-accept | High confidence |
| **0.75–0.85** | LLM validates | Gray zone — needs second opinion |
| **0.65–0.75** | Optional LLM | Weak match |
| **< 0.65** | Reject | Low confidence |

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | For OpenAI LLM | OpenAI API key |
| `ANTHROPIC_API_KEY` | For Anthropic LLM | Anthropic API key |

## Methodological Notes

**Strengths:**
- Semantic matching captures meaning beyond string overlap
- Geographic blocking reduces false positives and computation
- LLM validation provides expert-level judgment on edge cases
- Transparent, configurable thresholds

**Limitations:**
- Requires geographic data for blocking (falls back to full comparison without it)
- Entity rebranding may not be captured
- LLM validation adds cost and latency
- Embedding model quality depends on entity name characteristics

## License

MIT
