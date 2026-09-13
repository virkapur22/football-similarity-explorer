"""
load_fbref_kaggle.py — Ingest for players_data_light-2024_2025.csv

Real FBref scrape: attacking, passing, possession, defense, AND
goalkeeper stats all in one file. Handles two things the raw file
doesn't give you for free:

1. No player_id column — built from (Player name, Born year).
2. Players who changed clubs mid-season have TWO rows (one per club).
   We keep whichever row has the most minutes (their main stint) —
   note this slightly underrepresents a transferred player's full
   season, since the other club's minutes/goals/etc. are dropped.
"""

import pandas as pd


def load_fbref_kaggle_players(
    csv_path: str, column_map: dict[str, str], rate_columns: dict[str, str]
) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    df["player_id"] = df["Player"].astype(str) + "_" + df["Born"].astype(str)
    df["name"] = df["Player"]
    df["team"] = df["Squad"]
    df["league"] = df["Comp"]
    df["minutes"] = df["Min"]
    df["age"] = pd.to_numeric(df["Age"], errors="coerce")
    # FBref stores values like "fr FRA"; keep the readable three-letter code.
    df["nationality"] = df["Nation"].fillna("Unknown").astype(str).str.split().str[-1]
    # "DF,MF" -> "DF" (primary position; combo tells you they're versatile
    # but we need one group to search within)
    df["position"] = df["Pos"].astype(str).str.split(",").str[0]

    # keep each player's highest-minutes row (their main club that season)
    df = df.sort_values("minutes", ascending=False).drop_duplicates(
        subset="player_id", keep="first"
    )
    total_rows_before_dedupe = pd.read_csv(csv_path).shape[0]
    if total_rows_before_dedupe > len(df):
        print(
            f"[load_fbref_kaggle] collapsed {total_rows_before_dedupe} rows into "
            f"{len(df)} unique players (kept each player's highest-minutes club "
            f"when they changed clubs mid-season)"
        )

    rename = {raw: canon for raw, canon in column_map.items() if raw in df.columns}
    df = df.rename(columns=rename)
    rate_rename = {
        raw: canon for raw, canon in rate_columns.items() if raw in df.columns
    }
    df = df.rename(columns=rate_rename)

    keep_cols = [
        "player_id",
        "name",
        "position",
        "team",
        "league",
        "nationality",
        "age",
        "minutes",
    ]
    keep_cols += list(rename.values())
    keep_cols += list(rate_rename.values())
    df = df[keep_cols].copy()

    numeric_cols = [
        c
        for c in keep_cols
        if c not in ("player_id", "name", "position", "team", "league", "nationality")
    ]
    # Preserve missingness here. The cleaning stage imputes within positional
    # cohorts; treating every unavailable value as a football action count of
    # zero would distort both percentiles and vector geometry.
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")

    print(f"[load_fbref_kaggle] loaded {len(df)} unique players")
    return df


if __name__ == "__main__":
    import yaml

    config = yaml.safe_load(open("config_fbref_kaggle.yaml"))
    df = load_fbref_kaggle_players(
        "data/raw/fbref_kaggle/players_data_light-2024_2025.csv",
        config["column_map"],
        config["rate_columns"],
    )
    print(df.shape)
    print(df.head())
