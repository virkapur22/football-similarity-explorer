"""Position-aware player vectors and visualization projections."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


def family_weight_vector(
    dimension_columns: list[str], family: str, config: dict
) -> np.ndarray:
    configured = config.get("position_similarity_weights", {}).get(family, {})
    # sqrt is required so weighted cosine contributes exactly the configured
    # amount after the dot product rather than squaring the intended weight.
    return np.sqrt(
        np.array(
            [max(0.0, float(configured.get(name, 0.0))) for name in dimension_columns]
        )
    )


def display_projection(dimensions: pd.DataFrame, variance_target: float = 0.90) -> dict:
    """Fit full variance-targeted PCA plus a separate 3D display projection."""
    values = dimensions.to_numpy(float)
    full = PCA(n_components=variance_target, svd_solver="full").fit(values)
    display = PCA(n_components=3).fit(values)
    return {
        "analysis_pca": full,
        "display_pca": display,
        "coordinates": display.transform(values),
    }
