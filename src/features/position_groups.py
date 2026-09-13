"""
position_groups.py — Stage 3b: Feature engineering (position mapping)

Maps detailed scraped positions (LB, RB, LW, RW, ...) down to broad
groups (FB, W, ...) so similarity search can be restricted to
players in comparable roles.
"""

import pandas as pd


def add_position_group(
    df: pd.DataFrame, position_group_map: dict[str, str]
) -> pd.DataFrame:
    df = df.copy()
    df["position_group"] = df["position"].map(position_group_map)
    unmapped_mask = df["position_group"].isna()
    unmapped = unmapped_mask.sum()
    if unmapped:
        unmapped_values = df.loc[unmapped_mask, "position"].value_counts().head(15)
        print(
            f"[position_groups] warning: {unmapped} players had no mapping, "
            f"defaulting to their raw position. Unmapped codes seen:"
        )
        print(unmapped_values.to_string())
        df["position_group"] = df["position_group"].fillna(df["position"])
    return df
