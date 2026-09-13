"""
inspect_fbref_kaggle.py

Print the schema and a short sample from the FBref dataset used by the app.

Usage: python scripts/inspect_fbref_kaggle.py
"""

from pathlib import Path
import pandas as pd

RAW_DIR = Path("data/raw/fbref_kaggle")


def inspect_csv(name: str):
    path = RAW_DIR / name
    if not path.exists():
        print(f"!! {name} not found at {path}")
        return
    df = pd.read_csv(path)
    print(f"\n===== {name} =====")
    print(f"shape: {df.shape}")
    print("columns:", df.columns.tolist())
    print(df.head(3).to_string())


if __name__ == "__main__":
    inspect_csv("players_data_light-2024_2025.csv")
