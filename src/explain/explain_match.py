"""Explain player matches in the same dimension space used for similarity."""

import numpy as np
import pandas as pd


def explain_dimension_similarity(
    df: pd.DataFrame,
    dimension_matrix: np.ndarray,
    dimension_columns: list[str],
    name_a: str,
    name_b: str,
    top_common: int = 5,
    top_differences: int = 4,
) -> dict:
    """Explain a match with shared traits and material differences.

    This uses exactly the same football dimensions as the similarity engine;
    no generated prose or separate model is involved.
    """
    name_to_idx = {name.casefold(): i for i, name in enumerate(df["name"])}
    a, b = name_to_idx[name_a.casefold()], name_to_idx[name_b.casefold()]
    va, vb = dimension_matrix[a], dimension_matrix[b]
    closeness = -np.abs(va - vb)
    shared_strength = np.minimum(va, vb) + 0.5 * closeness
    active = np.where(np.maximum(np.abs(va), np.abs(vb)) > 1e-9)[0]
    common_order = active[np.argsort(shared_strength[active])[::-1][:top_common]]
    difference_order = active[
        np.argsort(np.abs(va[active] - vb[active]))[::-1][:top_differences]
    ]

    common = [
        {
            "dimension": dimension_columns[i],
            "label": dimension_columns[i].replace("_", " ").title(),
            "player_a": round(float(va[i]), 2),
            "player_b": round(float(vb[i]), 2),
        }
        for i in common_order
    ]
    differences = []
    for i in difference_order:
        leader = name_a if va[i] > vb[i] else name_b
        differences.append(
            {
                "dimension": dimension_columns[i],
                "label": dimension_columns[i].replace("_", " ").title(),
                "leader": leader,
                "gap": round(float(abs(va[i] - vb[i])), 2),
            }
        )
    return {
        "common_traits": common,
        "differences": differences,
        "reasons": [
            f"Both show a similar {item['label'].lower()} profile"
            for item in common[:3]
        ],
    }
