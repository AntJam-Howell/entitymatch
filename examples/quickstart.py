"""
Quick Start Example for entitymatch
====================================

Demonstrates matching two small company datasets using the full pipeline.
"""

import pandas as pd

from entitymatch import match_entities

# --- Create sample datasets ---

# Dataset A: Companies from one source
companies_a = pd.DataFrame(
    {
        "id": ["A001", "A002", "A003", "A004", "A005"],
        "company_name": [
            "McDonald's Corporation",
            "International Business Machines",
            "Walmart Inc",
            "Johnson & Johnson",
            "The Coca-Cola Company",
        ],
        "city": ["Chicago", "Armonk", "Bentonville", "New Brunswick", "Atlanta"],
        "state": ["IL", "NY", "AR", "NJ", "GA"],
    }
)

# Dataset B: Companies from another source (with name variations)
companies_b = pd.DataFrame(
    {
        "id": ["B001", "B002", "B003", "B004", "B005", "B006"],
        "company_name": [
            "McDonalds Corp",
            "IBM",
            "Wal-Mart Stores",
            "Johnson and Johnson Inc",
            "Coca Cola Co",
            "Target Corporation",
        ],
        "city": ["Chicago", "Armonk", "Bentonville", "New Brunswick", "Atlanta", "Minneapolis"],
        "state": ["IL", "NY", "AR", "NJ", "GA", "MN"],
    }
)

print("Dataset A:")
print(companies_a[["id", "company_name", "city", "state"]])
print()

print("Dataset B:")
print(companies_b[["id", "company_name", "city", "state"]])
print()

# --- Run matching (similarity only, no LLM) ---

print("Running entity matching...")
print()

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
    threshold=0.50,            # Lower threshold to see more candidates
    auto_accept_threshold=0.80,
    top_k=2,
    show_progress=False,
)

print("Match Results:")
print(results[["left_id", "right_id", "left_name", "right_name", "score", "accept_reason"]].to_string(index=False))
print()
print(f"Total accepted matches: {len(results)}")

# --- With LLM validation (requires API key) ---
#
# import os
# os.environ["OPENAI_API_KEY"] = "your-key-here"
#
# results_llm = match_entities(
#     df_left=companies_a,
#     df_right=companies_b,
#     left_name_col="company_name",
#     right_name_col="company_name",
#     left_id_col="id",
#     right_id_col="id",
#     left_city_col="city",
#     right_city_col="city",
#     left_state_col="state",
#     right_state_col="state",
#     use_llm=True,
#     llm_provider="openai",
#     llm_model="gpt-4o-mini",
# )
