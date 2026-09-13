"""FastAPI surface for player and archetype cosine similarity."""

from __future__ import annotations

import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from src.explain.explain_match import explain_dimension_similarity
from src.model.archetypes import archetype_to_api
from src.model.catalog import (
    FAMILY_COMPARISON_STATS,
    FAMILY_STAT_PRIORITIES,
    GOALKEEPER_STAT_KEYS,
    STAT_LABELS,
)
from src.model.pipeline import build_model

ROOT = Path(__file__).resolve().parents[2]
model = build_model(ROOT)
players = model.players
engine = model.engine
dimensions = model.dimensions
dimension_percentiles = model.dimension_percentiles
stat_percentiles = model.stat_percentiles
archetypes = model.archetypes
PROFILE_STAT_KEYS = [
    key
    for key in STAT_LABELS
    if key in players.columns and key in stat_percentiles.columns
]

app = FastAPI(title="Football Similarity Explorer API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _normalise(value: str) -> str:
    return "".join(
        character
        for character in unicodedata.normalize("NFKD", value.casefold())
        if not unicodedata.combining(character)
    )


def _search_rank(name: str, query: str) -> tuple[int, float, str]:
    normalised = _normalise(name)
    kind = (
        0
        if normalised.startswith(query)
        else (
            1
            if any(part.startswith(query) for part in normalised.split())
            else 2 if query in normalised else 3
        )
    )
    ratio = max(
        SequenceMatcher(None, query, normalised).ratio(),
        *(SequenceMatcher(None, query, part).ratio() for part in normalised.split()),
    )
    return kind, -ratio, normalised


def _percentile_stat(index: int, key: str) -> dict:
    label, unit = STAT_LABELS[key]
    return {
        "key": key,
        "label": label,
        "value": round(float(players.at[index, key]), 2),
        "unit": unit,
        "percentile": round(float(stat_percentiles.at[index, key]), 1),
    }


def _eligible_stat_keys(row) -> list[str]:
    if row["position_group"] == "GK":
        return [
            key
            for key in PROFILE_STAT_KEYS
            if key in GOALKEEPER_STAT_KEYS
            or key in {"pass_completion_pct", "passes_completed_p90"}
        ]
    return [key for key in PROFILE_STAT_KEYS if key not in GOALKEEPER_STAT_KEYS]


def _key_stats(row) -> list[dict]:
    priorities = FAMILY_STAT_PRIORITIES.get(row["position_family"], [])
    return [
        _percentile_stat(int(row.name), key)
        for key in priorities
        if key in PROFILE_STAT_KEYS
    ][:8]


def _all_stats(row) -> list[dict]:
    output = []
    cohort = players[players["position_group"] == row["position_group"]]
    for key in _eligible_stat_keys(row):
        stat = _percentile_stat(int(row.name), key)
        stat["scale_max"] = round(max(float(cohort[key].quantile(0.95)), 0.01), 2)
        output.append(stat)
    return output


def _archetype_profile(index: int) -> dict:
    matches = archetypes.matches_for_player(index)
    return {
        "primary": matches[0] if matches else None,
        "secondary": matches[1] if len(matches) > 1 else None,
        "matches": matches,
        "position_family": players.at[index, "position_family"],
        "note": "A cosine comparison with human-defined statistical prototype vectors. Scores describe profile resemblance, not performance or probability.",
        "display_scale": "raw cosine × 100, clamped to 0–100",
    }


def _player_profile(index: int) -> dict:
    row = players.iloc[index]
    eligible = [_percentile_stat(index, key) for key in _eligible_stat_keys(row)]
    ranked = sorted(eligible, key=lambda item: item["percentile"], reverse=True)
    family = row["position_family"]
    comparison_keys = [
        key
        for key in FAMILY_COMPARISON_STATS.get(family, [])
        if key in PROFILE_STAT_KEYS
    ]
    radar = [
        {
            "category": name.replace("_", " ").title(),
            "value": round(float(dimension_percentiles.at[index, name]), 1),
        }
        for name, weight in model.model_config["position_similarity_weights"]
        .get(family, {})
        .items()
        if weight > 0 and name in dimension_percentiles
    ]
    family_names = row["position_family_label"]
    confidence_note = (
        "hybrid statistical profile"
        if len(row["position_families"]) > 1
        else "primary statistical profile"
    )
    return {
        "name": row["name"],
        "team": row["team"],
        "league": row["league"],
        "nationality": row["nationality"],
        "age": int(row["age"]),
        "position": row["position"],
        "position_group": row["position_group"],
        "position_family": family,
        "position_families": row["position_families"],
        "position_family_label": family_names,
        "archetype": row["primary_archetype"],
        "primary_archetype": row["primary_archetype"],
        "minutes": int(row["minutes"]),
        "key_stats": _key_stats(row),
        "all_stats": _all_stats(row),
        "comparison_keys": comparison_keys,
        "radar": radar,
        "percentile_stats": ranked,
        "strengths": ranked[:3],
        "development_areas": sorted(ranked, key=lambda item: item["percentile"])[:3],
        "playstyle_description": f"{family_names} — a {confidence_note} inferred from position-relative 2024/25 output, not a listed FBref position.",
        "stats_note": "Rates are per 90 minutes unless marked as a percentage. Percentiles compare players in the same broad position group.",
        "archetype_similarity": _archetype_profile(index),
    }


def _top_archetype_players(archetype_id: str, limit: int) -> list[dict]:
    definition = archetypes.definitions[archetype_id]
    scores = (
        archetypes.raw_cosines[archetype_id]
        .dropna()
        .sort_values(ascending=False)
        .head(limit)
    )
    return [
        {
            "name": players.at[index, "name"],
            "team": players.at[index, "team"],
            "league": players.at[index, "league"],
            "position_family": players.at[index, "position_family"],
            "similarity": round(float(max(0.0, score) * 100.0), 1),
            "cosine_similarity": round(float(score), 4),
        }
        for index, score in scores.items()
        if players.at[index, "position_family"] in definition["families"]
    ]


def _archetype_payload(archetype_id: str, top_n: int = 5) -> dict:
    definition = archetypes.definitions[archetype_id]
    payload = archetype_to_api(archetype_id, definition)
    payload["top_players"] = _top_archetype_players(archetype_id, top_n)
    payload["eligible_player_count"] = int(
        archetypes.raw_cosines[archetype_id].notna().sum()
    )
    payload["target_vectors"] = {
        family: [
            {
                "feature": feature,
                "normalized_target": round(float(value), 4),
                "target_percentile": float(definition["target_percentiles"][feature]),
            }
            for feature, value in zip(
                definition["features"],
                archetypes.target_vectors[(archetype_id, family)],
            )
        ]
        for family in definition["families"]
        if (archetype_id, family) in archetypes.target_vectors
    }
    valid = archetypes.similarities[archetype_id].dropna()
    payload["distribution"] = {
        "minimum": round(float(valid.min()), 1),
        "median": round(float(valid.median()), 1),
        "p90": round(float(valid.quantile(0.90)), 1),
        "maximum": round(float(valid.max()), 1),
    }
    return payload


@app.get("/players")
def list_players():
    return players[
        [
            "name",
            "team",
            "league",
            "position_group",
            "position_family",
            "position_family_label",
            "primary_archetype",
        ]
    ].to_dict(orient="records")


@app.get("/search")
def search_players(q: str = Query(default="", max_length=100)):
    query = _normalise(q).strip()
    if not query:
        return []
    results = players[["name", "position", "position_group", "team", "league"]].copy()
    results["_normalised"] = results["name"].map(_normalise)
    results["_rank"] = results["name"].map(lambda name: _search_rank(name, query))
    direct = results["_normalised"].str.contains(query, regex=False)
    if direct.any():
        results = results[direct]
    elif len(query) >= 3:
        results = results[results["_rank"].map(lambda rank: -rank[1]) >= 0.70]
    else:
        return []
    return (
        results.sort_values("_rank", kind="stable")
        .head(10)
        .drop(columns=["_normalised", "_rank"])
        .to_dict(orient="records")
    )


@app.get("/player/{player_name}")
def player_profile(player_name: str):
    try:
        index = engine.resolve_index(player_name)
    except ValueError as error:
        raise HTTPException(status_code=404, detail=str(error))
    profile = _player_profile(index)
    similar = engine.find_similar(player_name, n=5, same_position_only=True)
    profile["similar_players"] = [
        {
            "name": item["name"],
            "team": players.loc[players["name"] == item["name"], "team"].iloc[0],
            "position_group": item["position_group"],
            "position_family": item.get("position_family"),
            "similarity": float(item["similarity"]),
            "cosine_similarity": float(item["cosine_similarity"]),
        }
        for item in similar.to_dict(orient="records")
    ]
    return profile


@app.get("/archetypes")
def archetype_catalog(category: str = "", top_n: int = Query(default=5, ge=0, le=20)):
    items = [
        _archetype_payload(key, top_n)
        for key, definition in archetypes.definitions.items()
        if not category or definition["category"].casefold() == category.casefold()
    ]
    return {
        "categories": list(
            dict.fromkeys(
                definition["category"] for definition in archetypes.definitions.values()
            )
        ),
        "archetypes": items,
        "method": "Human-defined percentile targets are converted into the same robust-scaled dimensions as eligible players, then compared with cosine similarity.",
        "display_scale": "raw cosine × 100, clamped to 0–100; similarity is not performance or probability",
    }


@app.get("/archetypes/{archetype_id}")
def archetype_detail(archetype_id: str, top_n: int = Query(default=10, ge=1, le=50)):
    if archetype_id not in archetypes.definitions:
        raise HTTPException(
            status_code=404, detail=f"Unknown archetype: {archetype_id}"
        )
    return _archetype_payload(archetype_id, top_n)


@app.get("/archetype-similarity/{player_name}")
def player_archetype_similarity(player_name: str):
    try:
        index = engine.resolve_index(player_name)
    except ValueError as error:
        raise HTTPException(status_code=404, detail=str(error))
    return {"player": players.at[index, "name"], **_archetype_profile(index)}


def _parse_thresholds(value: str) -> dict[str, float]:
    output = {}
    for item in value.split(","):
        if ":" not in item:
            continue
        key, raw = item.split(":", 1)
        if key in PROFILE_STAT_KEYS:
            try:
                output[key] = min(100.0, max(0.0, float(raw)))
            except ValueError:
                pass
    return output


def _archetype_display_keys(archetype_id: str, row, sort_by: str) -> list[str]:
    keys = (
        list(archetypes.definitions[archetype_id]["display_metrics"])
        if archetype_id
        else []
    )
    if sort_by in PROFILE_STAT_KEYS:
        keys.insert(0, sort_by)
    keys.extend(FAMILY_STAT_PRIORITIES.get(row["position_family"], []))
    eligible = set(_eligible_stat_keys(row))
    return list(dict.fromkeys(key for key in keys if key in eligible))[:3]


@app.get("/archetype-options")
def archetype_search_options():
    items = []
    for key, definition in archetypes.definitions.items():
        item = archetype_to_api(key, definition)
        item["positions"] = sorted(
            {
                (
                    "GK"
                    if family == "GK"
                    else (
                        "DF"
                        if family in {"CB", "FB_WB"}
                        else "MF" if family in {"DM", "CM", "AM"} else "FW"
                    )
                )
                for family in definition["families"]
            }
        )
        items.append(item)
    return {
        "clubs": sorted(players["team"].dropna().astype(str).unique()),
        "leagues": sorted(players["league"].dropna().astype(str).unique()),
        "nationalities": sorted(players["nationality"].dropna().astype(str).unique()),
        "positions": sorted(players["position_group"].dropna().astype(str).unique()),
        "players": players[["name", "team"]]
        .sort_values("name")
        .to_dict(orient="records"),
        "stats": [
            {"key": key, "label": STAT_LABELS[key][0], "unit": STAT_LABELS[key][1]}
            for key in PROFILE_STAT_KEYS
        ],
        "archetypes": items,
        "percentile_basis": "Each statistic is ranked only against players in the same broad group (GK, DF, MF or FW).",
    }


@app.get("/archetype-search")
def search_by_archetype(
    q: str = "",
    club: str = "",
    league: str = "",
    nationality: str = "",
    position: str = "",
    archetype: str = "",
    min_age: int = Query(default=0, ge=0, le=60),
    max_age: int = Query(default=60, ge=0, le=60),
    min_minutes: int = Query(default=900, ge=0),
    thresholds: str = "",
    sort_by: str = "archetype_similarity",
    sort_order: str = "desc",
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=10, ge=1, le=50),
    limit: int | None = Query(default=None, ge=1, le=150),
):
    if archetype and archetype not in archetypes.definitions:
        raise HTTPException(status_code=404, detail=f"Unknown archetype: {archetype}")
    mask = (
        (players["age"] >= min_age)
        & (players["age"] <= max_age)
        & (players["minutes"] >= min_minutes)
    )
    if q:
        mask &= players["name"].map(_normalise).str.contains(_normalise(q), regex=False)
    for column, value in (
        ("team", club),
        ("league", league),
        ("nationality", nationality),
        ("position_group", position),
    ):
        if value:
            mask &= players[column].str.casefold() == value.casefold()
    if archetype:
        mask &= players["position_family"].isin(
            archetypes.definitions[archetype]["families"]
        )
    for key, threshold in _parse_thresholds(thresholds).items():
        mask &= stat_percentiles[key] >= threshold

    results = []
    for index in players.index[mask]:
        row = players.loc[index]
        raw = float(archetypes.raw_cosines.at[index, archetype]) if archetype else None
        rounded_raw = round(raw, 4) if raw is not None and np.isfinite(raw) else None
        score = (
            round(max(0.0, rounded_raw) * 100.0, 1) if rounded_raw is not None else None
        )
        results.append(
            {
                "name": row["name"],
                "team": row["team"],
                "league": row["league"],
                "nationality": row["nationality"],
                "age": int(row["age"]),
                "position": row["position"],
                "position_group": row["position_group"],
                "position_family": row["position_family"],
                "minutes": int(row["minutes"]),
                "primary_archetype": row["primary_archetype"],
                "selected_archetype": archetype,
                "archetype_similarity": score,
                "cosine_similarity": rounded_raw,
                "stats": [
                    _percentile_stat(int(index), key)
                    for key in _archetype_display_keys(archetype, row, sort_by)
                ],
            }
        )
    reverse = sort_order.casefold() != "asc"
    if sort_by in PROFILE_STAT_KEYS:
        results.sort(
            key=lambda item: float(
                players.loc[players["name"] == item["name"], sort_by].iloc[0]
            ),
            reverse=reverse,
        )
    elif sort_by in {"age", "minutes", "name"}:
        results.sort(key=lambda item: item[sort_by], reverse=reverse)
    elif archetype:
        results.sort(
            key=lambda item: (
                item["archetype_similarity"]
                if item["archetype_similarity"] is not None
                else -1
            ),
            reverse=reverse,
        )
    else:
        results.sort(key=lambda item: item["minutes"], reverse=True)
    # ``limit`` remains as a compatibility alias for older API consumers.
    effective_page_size = limit or page_size
    result_count = len(results)
    page_count = max(1, (result_count + effective_page_size - 1) // effective_page_size)
    current_page = min(page, page_count)
    start = (current_page - 1) * effective_page_size
    return {
        "count": result_count,
        "page": current_page,
        "page_size": effective_page_size,
        "page_count": page_count,
        "has_previous": current_page > 1,
        "has_next": current_page < page_count,
        "results": results[start : start + effective_page_size],
        "score_label": "Archetype similarity",
        "percentile_basis": "Position-relative output; archetype score is raw cosine × 100 in the shared robust-scaled vector space.",
    }


@app.get("/similar/{player_name}")
def similar_players(
    player_name: str,
    n: int = Query(default=5, ge=1, le=50),
    same_position_only: bool = True,
):
    try:
        return engine.find_similar(
            player_name, n=n, same_position_only=same_position_only
        ).to_dict(orient="records")
    except ValueError as error:
        raise HTTPException(status_code=404, detail=str(error))


@app.get("/map/{player_name}")
def similarity_map(
    player_name: str,
    n: int = Query(default=5, ge=1, le=50),
    same_position_only: bool = True,
):
    try:
        target_idx, candidates, cosine = engine.score_candidates(
            player_name, same_position_only=same_position_only
        )
    except ValueError as error:
        raise HTTPException(status_code=404, detail=str(error))
    order = np.argsort(cosine)[::-1][:n]
    selected, raw_scores = candidates[order], cosine[order]
    display_scores = np.clip(raw_scores * 100.0, 0.0, 100.0)
    display_by_index = {
        int(index): round(float(score), 1)
        for index, score in zip(selected, display_scores)
    }
    cosine_by_index = {
        int(index): round(float(score), 4) for index, score in zip(selected, raw_scores)
    }
    selected_set = set(display_by_index)
    points = []
    for index, row in players.iterrows():
        points.append(
            {
                "name": row["name"],
                "team": row["team"],
                "league": row["league"],
                "position_group": row["position_group"],
                "position_family": row["position_family"],
                "archetype": row["primary_archetype"],
                "x": round(float(model.coordinates[index, 0]), 5),
                "y": round(float(model.coordinates[index, 1]), 5),
                "z": round(float(model.coordinates[index, 2]), 5),
                "is_target": index == target_idx,
                "is_similar": index in selected_set,
                "similarity": display_by_index.get(index),
                "cosine_similarity": cosine_by_index.get(index),
            }
        )
    target = points[target_idx]
    target["key_stats"] = _key_stats(players.iloc[target_idx])
    return {
        "target": target,
        "players": points,
        "connections": [
            {
                "from": target["name"],
                "to": points[int(index)]["name"],
                "similarity": display_by_index[int(index)],
                "cosine_similarity": cosine_by_index[int(index)],
            }
            for index in selected
        ],
        "pca": {
            "axes": ["PC1", "PC2", "PC3"],
            "explained_variance_ratio": [
                round(float(value), 5)
                for value in model.display_pca.explained_variance_ratio_
            ],
            "total_explained_variance": round(
                float(model.display_pca.explained_variance_ratio_.sum()), 5
            ),
            "analysis_components_for_90_percent": int(model.analysis_pca.n_components_),
        },
        "similarity": {
            "metric": "position-family weighted cosine",
            "feature_space": "robust-scaled football dimensions within broad position groups",
            "raw_scale": "cosine from -1 to 1",
            "display_scale": "raw cosine × 100 (clamped to 0–100%)",
            "same_position_only": same_position_only,
        },
    }


@app.get("/explain")
def explain(player_a: str, player_b: str):
    try:
        detail = explain_dimension_similarity(
            players,
            dimensions.to_numpy(float),
            model.dimension_columns,
            player_a,
            player_b,
        )
    except KeyError as error:
        raise HTTPException(status_code=404, detail=f"Player not found: {error}")
    return {"player_a": player_a, "player_b": player_b, **detail}


app.mount("/", StaticFiles(directory=ROOT / "frontend", html=True), name="frontend")
