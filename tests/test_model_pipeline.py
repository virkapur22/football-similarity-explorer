import numpy as np
from fastapi.testclient import TestClient
from sklearn.metrics.pairwise import cosine_similarity

from src.api.app import app, archetypes, dimensions, engine, players
from src.model.archetypes import ARCHETYPE_DEFINITIONS, _target_vector


client = TestClient(app)


def test_typeahead_returns_clean_player_name():
    response = client.get("/search", params={"q": "Kylian"})
    assert response.status_code == 200
    assert response.json()[0]["name"] == "Kylian Mbappé"


def test_outfield_profile_excludes_goalkeeper_statistics():
    profile = client.get("/player/Erling Haaland").json()
    keys = {stat["key"] for stat in profile["percentile_stats"]}
    assert "save_pct" not in keys
    assert "goals_prevented_p90" not in keys
    assert "quality" not in profile


def test_archetype_vector_uses_declared_feature_order():
    for key, definition in ARCHETYPE_DEFINITIONS.items():
        assert tuple(definition["target_percentiles"]) == definition["features"], key
        for family in definition["families"]:
            assert len(archetypes.target_vectors[(key, family)]) == len(
                definition["features"]
            )


def test_target_percentiles_are_converted_in_normalized_dimension_space():
    definition = ARCHETYPE_DEFINITIONS["poacher-box-striker"]
    indices = players.index[players["position_family"] == "ST"]
    values = dimensions.loc[indices, list(definition["features"])]
    expected = _target_vector(
        values, definition["features"], definition["target_percentiles"]
    )
    np.testing.assert_allclose(
        archetypes.target_vectors[("poacher-box-striker", "ST")], expected
    )


def test_archetype_similarity_is_direct_cosine():
    index = engine.resolve_index("Erling Haaland")
    definition = ARCHETYPE_DEFINITIONS["poacher-box-striker"]
    features = list(definition["features"])
    player_vector = dimensions.loc[index, features].to_numpy(float).reshape(1, -1)
    target_vector = archetypes.target_vectors[("poacher-box-striker", "ST")].reshape(
        1, -1
    )
    expected = float(cosine_similarity(player_vector, target_vector)[0, 0])
    assert archetypes.raw_cosines.at[index, "poacher-box-striker"] == expected


def test_position_eligibility_excludes_irrelevant_archetypes():
    index = engine.resolve_index("Erling Haaland")
    assert np.isnan(archetypes.raw_cosines.at[index, "shot-stopper"])
    index = engine.resolve_index("David Raya")
    assert np.isnan(archetypes.raw_cosines.at[index, "poacher-box-striker"])


def test_primary_archetype_is_highest_eligible_cosine():
    profile = client.get("/archetype-similarity/Erling Haaland").json()
    assert profile["primary"] == max(
        profile["matches"], key=lambda item: item["cosine_similarity"]
    )
    assert (
        profile["primary"]["name"]
        == players.loc[engine.resolve_index("Erling Haaland"), "primary_archetype"]
    )


def test_archetype_search_sorts_by_similarity():
    payload = client.get(
        "/archetype-search", params={"archetype": "poacher-box-striker", "limit": 20}
    ).json()
    scores = [row["archetype_similarity"] for row in payload["results"]]
    assert scores == sorted(scores, reverse=True)
    assert all(row["position_family"] == "ST" for row in payload["results"])


def test_archetype_search_uses_profile_specific_display_metrics():
    row = client.get(
        "/archetype-search", params={"archetype": "false-nine", "limit": 1}
    ).json()["results"][0]
    assert [stat["key"] for stat in row["stats"]] == [
        "xag_p90",
        "key_passes_p90",
        "progressive_passes_p90",
    ]


def test_archetype_search_paginates_ten_players():
    first = client.get("/archetype-search", params={"page": 1, "page_size": 10}).json()
    second = client.get("/archetype-search", params={"page": 2, "page_size": 10}).json()
    assert len(first["results"]) == len(second["results"]) == 10
    assert first["page"] == 1 and second["page"] == 2
    assert first["page_size"] == second["page_size"] == 10
    assert first["page_count"] == second["page_count"] > 1
    assert first["has_previous"] is False and first["has_next"] is True
    assert second["has_previous"] is True
    assert {row["name"] for row in first["results"]}.isdisjoint(
        row["name"] for row in second["results"]
    )


def test_catalog_is_transparent_and_human_defined():
    payload = client.get("/archetypes").json()
    assert len(payload["archetypes"]) == 25
    assert all(
        item["features"] and item["technical_description"]
        for item in payload["archetypes"]
    )
    assert "Human-defined" in payload["method"]


def test_player_similarity_remains_raw_cosine_times_100():
    rows = client.get("/similar/Luis Díaz", params={"n": 20}).json()
    dembele = next(row for row in rows if row["name"] == "Ousmane Dembélé")
    assert dembele["cosine_similarity"] == 0.9742
    assert dembele["similarity"] == 97.4


def test_legacy_role_cluster_and_quality_routes_are_removed():
    assert client.get("/roles").status_code == 404
    assert client.get("/role-fit/Erling Haaland/poacher").status_code == 404
    assert client.get("/scout").status_code == 404
    assert client.get("/clusters/diagnostics").status_code == 404
    assert client.get("/quality/diagnostics").status_code == 404


def test_approved_scout_ui_structure_is_preserved():
    html = client.get("/").text
    for element_id in (
        "scout-view",
        "scout-form",
        "scout-role",
        "scout-results",
        "sort-direction",
        "shortlist-tray",
    ):
        assert f'id="{element_id}"' in html
    assert "Archetype similarity" in html
