"""Build and inspect the same model used by the API.

Usage: python main_fbref_kaggle.py [player name]
"""

import sys
from pathlib import Path

from src.model.pipeline import build_model

ROOT = Path(__file__).resolve().parent


def main() -> None:
    model = build_model(ROOT)
    player = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Erling Haaland"
    index = model.engine.resolve_index(player)
    row = model.players.iloc[index]
    print(
        f"{row['name']} · {row['team']} · {row['position_group']} · {row['archetype']}"
    )
    print(model.engine.find_similar(player, n=5).to_string(index=False))
    print(f"\n90% PCA components: {model.analysis_pca.n_components_}")


if __name__ == "__main__":
    main()
