"""Single source of truth for the football modelling pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.clean.clean_stats import clean_players
from src.features.football_dimensions import (
    build_football_dimensions,
    infer_position_families,
)
from src.features.per90 import add_per90_columns
from src.features.position_groups import add_position_group
from src.ingest.load_fbref_kaggle import load_fbref_kaggle_players
from src.model.archetypes import ArchetypeArtifacts, build_archetype_artifacts
from src.model.player_vectors import display_projection, family_weight_vector
from src.model.similarity import PlayerSimilarityEngine


STYLE_LABELS = {
    "GK": "Goalkeeper",
    "CB": "Centre-back",
    "FB_WB": "Full-back / wing-back",
    "DM": "Defensive midfielder",
    "CM": "Central midfielder",
    "AM": "Attacking midfielder",
    "W": "Winger",
    "ST": "Striker",
}


@dataclass
class ModelArtifacts:
    players: pd.DataFrame
    dimensions: pd.DataFrame
    dimension_columns: list[str]
    dimension_percentiles: pd.DataFrame
    stat_percentiles: pd.DataFrame
    engine: PlayerSimilarityEngine
    coordinates: np.ndarray
    display_pca: object
    analysis_pca: object
    data_config: dict
    model_config: dict
    archetypes: ArchetypeArtifacts


def build_model(root: Path) -> ModelArtifacts:
    data_config = yaml.safe_load((root / "config_fbref_kaggle.yaml").read_text())
    model_config = yaml.safe_load((root / "config_model.yaml").read_text())
    raw = load_fbref_kaggle_players(
        str(root / "data/raw/fbref_kaggle/players_data_light-2024_2025.csv"),
        data_config["column_map"],
        data_config["rate_columns"],
    )
    players = add_position_group(
        add_per90_columns(
            clean_players(raw, data_config["min_minutes"]),
            list(data_config["column_map"].values()),
        ),
        data_config["position_group_map"],
    ).reset_index(drop=True)
    dimensions, dimension_columns, dimension_percentiles, _ = build_football_dimensions(
        players, model_config["dimensions"], model_config["scaling"]
    )
    families = infer_position_families(players, dimensions)
    players = pd.concat([players, families], axis=1)
    players["position_family_label"] = players["position_families"].map(
        lambda values: " / ".join(STYLE_LABELS.get(value, value) for value in values)
    )
    stat_columns = [
        column
        for column in players
        if column.endswith("_p90")
        or column.endswith("_pct")
        or column
        in {"npxg_per_shot", "avg_defensive_action_distance", "gk_average_pass_length"}
    ]
    stat_percentiles = (
        players.groupby("position_group")[stat_columns]
        .rank(method="average", pct=True)
        .mul(100)
    )
    weights = {
        family: family_weight_vector(dimension_columns, family, model_config)
        for family in model_config["position_similarity_weights"]
    }
    engine = PlayerSimilarityEngine(
        players, dimensions.to_numpy(float), dimension_columns, weights
    )
    projection = display_projection(
        dimensions, float(model_config["pca"]["variance_target"])
    )
    archetypes = build_archetype_artifacts(players, dimensions)
    players["primary_archetype"] = [
        (
            matches[0]["name"]
            if (matches := archetypes.matches_for_player(int(index)))
            else "Unclassified"
        )
        for index in players.index
    ]
    # Compatibility field used by the approved Explore UI.
    players["archetype"] = players["primary_archetype"]
    return ModelArtifacts(
        players=players,
        dimensions=dimensions,
        dimension_columns=dimension_columns,
        dimension_percentiles=dimension_percentiles,
        stat_percentiles=stat_percentiles,
        engine=engine,
        coordinates=projection["coordinates"],
        display_pca=projection["display_pca"],
        analysis_pca=projection["analysis_pca"],
        data_config=data_config,
        model_config=model_config,
        archetypes=archetypes,
    )
