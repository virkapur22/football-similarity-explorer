"""Football-aware feature engineering shared by every product surface.

The raw FBref columns are deliberately collapsed into a smaller set of
interpretable dimensions.  This reduces double-counting from highly correlated
box-score statistics and gives Explore, Scout and reports the same vocabulary.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler


OUTFIELD_DIMENSIONS = {
    "distribution",
    "creativity",
    "progression",
    "dribbling",
    "goal_threat",
    "finishing",
    "defensive_activity",
    "defensive_aggression",
    "aerial",
    "movement",
    "ball_security",
}
GOALKEEPER_DIMENSIONS = {
    "shot_stopping",
    "gk_distribution",
    "gk_sweeping",
    "ball_security",
}


def robust_scale_by_position(
    df: pd.DataFrame,
    feature_columns: list[str],
    quantile_range: tuple[float, float] = (10.0, 90.0),
    clip: float = 4.0,
) -> tuple[pd.DataFrame, dict[str, RobustScaler]]:
    """Robust-scale raw inputs inside GK/DF/MF/FW cohorts.

    Scaling inside each cohort prevents structural goalkeeper zeros from
    affecting outfield vectors and limits the leverage of small-sample outliers.
    """
    scaled = pd.DataFrame(0.0, index=df.index, columns=feature_columns)
    scalers: dict[str, RobustScaler] = {}
    for group, indices in df.groupby("position_group").groups.items():
        scaler = RobustScaler(quantile_range=quantile_range)
        values = (
            df.loc[indices, feature_columns]
            .astype(float)
            .replace([np.inf, -np.inf], np.nan)
        )
        values = values.fillna(values.median()).fillna(0.0)
        transformed = np.clip(scaler.fit_transform(values), -clip, clip)
        scaled.loc[indices, feature_columns] = transformed
        scalers[str(group)] = scaler
    return scaled, scalers


def build_football_dimensions(
    df: pd.DataFrame,
    dimension_config: dict[str, dict[str, float]],
    scaling_config: dict,
) -> tuple[pd.DataFrame, list[str], pd.DataFrame, dict[str, RobustScaler]]:
    """Return dimension scores and their position-relative percentiles."""
    raw_features = list(
        dict.fromkeys(
            feature
            for members in dimension_config.values()
            for feature in members
            if feature in df.columns
        )
    )
    scaled_raw, scalers = robust_scale_by_position(
        df,
        raw_features,
        tuple(scaling_config.get("quantile_range", [10, 90])),
        float(scaling_config.get("clip", 4.0)),
    )
    dimensions = pd.DataFrame(index=df.index)
    for name, members in dimension_config.items():
        available = {
            key: float(weight)
            for key, weight in members.items()
            if key in scaled_raw.columns
        }
        if not available:
            dimensions[name] = 0.0
            continue
        denominator = sum(abs(weight) for weight in available.values()) or 1.0
        dimensions[name] = (
            sum(scaled_raw[key] * weight for key, weight in available.items())
            / denominator
        )

    # Structural missingness stays out of the model instead of becoming a
    # shared artificial signal among outfield players or goalkeepers.
    outfield = df["position_group"] != "GK"
    dimensions.loc[outfield, list(GOALKEEPER_DIMENSIONS - {"ball_security"})] = 0.0
    dimensions.loc[~outfield, list(OUTFIELD_DIMENSIONS - {"ball_security"})] = 0.0

    dimension_columns = list(dimension_config)
    percentiles = (
        dimensions.groupby(df["position_group"])[dimension_columns]
        .rank(method="average", pct=True)
        .mul(100)
    )
    return dimensions, dimension_columns, percentiles, scalers


def infer_position_families(
    df: pd.DataFrame,
    dimensions: pd.DataFrame,
    hybrid_margin: float = 0.40,
) -> pd.DataFrame:
    """Infer detailed role families where FBref supplies only broad positions.

    These are transparent statistical inferences, not ground-truth positions.
    A player receives a second family when the two best scores are close.
    """

    def d(index: int, key: str) -> float:
        return float(dimensions.at[index, key]) if key in dimensions else 0.0

    output = []
    for index, row in df.iterrows():
        broad = row["position_group"]
        if broad == "GK":
            scores = {"GK": 1.0}
        elif broad == "DF":
            scores = {
                "CB": 0.36 * d(index, "defensive_activity")
                + 0.27 * d(index, "aerial")
                + 0.20 * d(index, "ball_security")
                + 0.17 * d(index, "distribution"),
                "FB_WB": 0.29 * d(index, "progression")
                + 0.24 * d(index, "dribbling")
                + 0.20 * d(index, "creativity")
                + 0.15 * d(index, "movement")
                + 0.12 * d(index, "defensive_activity"),
            }
        elif broad == "MF":
            scores = {
                "DM": 0.36 * d(index, "defensive_activity")
                + 0.23 * d(index, "ball_security")
                + 0.22 * d(index, "distribution")
                + 0.19 * d(index, "defensive_aggression"),
                "CM": 0.27 * d(index, "distribution")
                + 0.27 * d(index, "progression")
                + 0.18 * d(index, "movement")
                + 0.15 * d(index, "defensive_activity")
                + 0.13 * d(index, "creativity"),
                "AM": 0.34 * d(index, "creativity")
                + 0.22 * d(index, "dribbling")
                + 0.20 * d(index, "goal_threat")
                + 0.16 * d(index, "movement")
                + 0.08 * d(index, "progression"),
            }
        else:
            scores = {
                "W": 0.31 * d(index, "dribbling")
                + 0.25 * d(index, "creativity")
                + 0.18 * d(index, "progression")
                + 0.15 * d(index, "movement")
                + 0.11 * d(index, "goal_threat"),
                "ST": 0.34 * d(index, "goal_threat")
                + 0.27 * d(index, "finishing")
                + 0.19 * d(index, "movement")
                + 0.13 * d(index, "aerial")
                + 0.07 * d(index, "ball_security"),
            }
        ranked = sorted(scores, key=scores.get, reverse=True)
        families = [ranked[0]]
        if len(ranked) > 1 and scores[ranked[0]] - scores[ranked[1]] <= hybrid_margin:
            families.append(ranked[1])
        spread = scores[ranked[0]] - scores[ranked[1]] if len(ranked) > 1 else 1.0
        output.append(
            {
                "position_family": ranked[0],
                "position_families": families,
                "position_family_confidence": round(
                    float(1 / (1 + np.exp(-2 * spread))), 3
                ),
            }
        )
    return pd.DataFrame(output, index=df.index)
