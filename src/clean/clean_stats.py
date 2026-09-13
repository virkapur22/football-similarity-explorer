"""
clean_stats.py — Stage 2: Clean

Filters low-minute samples (noisy per-90 stats), drops duplicate
player-seasons, and fills gaps. Keep this stage dumb and mechanical —
no feature engineering here, that's the next stage.
"""

import pandas as pd


def clean_players(df: pd.DataFrame, min_minutes: int = 450) -> pd.DataFrame:
    df = df.copy()

    # drop players who haven't played enough for per-90 stats to be meaningful
    before = len(df)
    df = df[df["minutes"] >= min_minutes]
    dropped = before - len(df)
    if dropped:
        print(f"[clean_stats] dropped {dropped} players below {min_minutes} minutes")

    # keep the highest-minutes row if a player_id appears more than once
    df = df.sort_values("minutes", ascending=False).drop_duplicates(
        subset="player_id", keep="first"
    )

    # Impute genuine gaps from positional peers. A zero fallback is reserved
    # for columns unavailable to an entire position (structural missingness,
    # such as goalkeeper-only measures for outfield players).
    numeric_cols = df.select_dtypes(include="number").columns
    for column in numeric_cols:
        positional_median = df.groupby("position")[column].transform("median")
        df[column] = df[column].fillna(positional_median)
    df[numeric_cols] = df[numeric_cols].fillna(0)

    return df.reset_index(drop=True)
