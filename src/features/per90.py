"""
per90.py — Stage 3a: Feature engineering (rate conversion)

Converts raw counting totals into per-90-minute rates so players
with different amounts of playing time are comparable.
"""

import pandas as pd


def add_per90_columns(df: pd.DataFrame, per90_stats: list[str]) -> pd.DataFrame:
    df = df.copy()
    for stat in per90_stats:
        df[f"{stat}_p90"] = (df[stat] / df["minutes"]) * 90
    return df
