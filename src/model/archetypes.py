"""Human-defined football archetype vectors compared with player vectors.

Archetypes are statistical prototypes, not ratings or learned classes.  Each
prototype declares a small, interpretable subset of the shared football
dimensions and a deliberate percentile target for every included dimension.
Those percentile targets are converted through the observed distribution of
the eligible position family, placing the prototype in exactly the same
robust-scaled coordinate system as the player vectors.  Similarity is then a
plain cosine between the player and prototype vectors.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


FAMILY_LABELS = {
    "GK": "Goalkeepers",
    "CB": "Centre backs",
    "FB_WB": "Fullbacks / wingbacks",
    "DM": "Midfielders",
    "CM": "Midfielders",
    "AM": "Midfielders",
    "W": "Wide attackers",
    "ST": "Forwards",
}


def _prototype(
    name: str,
    category: str,
    families: tuple[str, ...],
    description: str,
    targets: dict[str, float],
    rationale: dict[str, str],
) -> dict:
    return {
        "name": name,
        "category": category,
        "families": families,
        "description": description,
        "features": tuple(targets),
        "target_percentiles": targets,
        "rationale": rationale,
    }


# Percentile targets are intentionally explicit.  They describe the desired
# shape within an eligible position family; they are not coefficients and are
# never summed into a score.
ARCHETYPE_DEFINITIONS: dict[str, dict] = {
    "poacher-box-striker": _prototype(
        "Poacher / Box Striker",
        "Forwards",
        ("ST",),
        "Lives close to goal, repeatedly reaches finishing positions and converts chances.",
        {"goal_threat": 95, "finishing": 95, "movement": 82, "aerial": 58},
        {
            "goal_threat": "Elite shot and penalty-area presence defines the profile.",
            "finishing": "High conversion and shot accuracy separate finishers from volume shooters.",
            "movement": "Strong receiving and box occupation support repeated chances.",
            "aerial": "Useful but not mandatory, so the target is only moderately above average.",
        },
    ),
    "target-forward": _prototype(
        "Target Forward",
        "Forwards",
        ("ST",),
        "Provides an aerial outlet, occupies centre backs and remains a meaningful scoring threat.",
        {"aerial": 95, "goal_threat": 82, "movement": 72, "ball_security": 68},
        {
            "aerial": "Aerial volume and success are the defining signal.",
            "goal_threat": "The outlet must still threaten the box.",
            "movement": "Receiving involvement represents availability as a focal point.",
            "ball_security": "Retention supports hold-up play without turning it into a performance rating.",
        },
    ),
    "complete-forward": _prototype(
        "Complete Forward",
        "Forwards",
        ("ST",),
        "Combines scoring, carrying, creation, movement and physical presence.",
        {
            "goal_threat": 88,
            "finishing": 86,
            "movement": 84,
            "dribbling": 78,
            "creativity": 76,
            "aerial": 74,
        },
        {
            "goal_threat": "Sustained scoring threat remains central.",
            "finishing": "Above-average conversion complements volume.",
            "movement": "High involvement keeps the forward connected.",
            "dribbling": "Carrying adds self-created threat.",
            "creativity": "Chance creation distinguishes an all-phase forward.",
            "aerial": "Physical variety is expected, though not at target-forward level.",
        },
    ),
    "false-nine": _prototype(
        "False 9 / Creative Forward",
        "Forwards",
        ("ST",),
        "Drops into connective areas, creates for others and carries through pressure.",
        {
            "creativity": 94,
            "distribution": 86,
            "ball_security": 84,
            "dribbling": 82,
            "movement": 78,
            "goal_threat": 62,
        },
        {
            "creativity": "Creation is the clearest contrast with a box striker.",
            "distribution": "Passing involvement represents dropping into build-up.",
            "ball_security": "Secure possession supports combination play.",
            "dribbling": "Ball carrying helps connect midfield and attack.",
            "movement": "Receiving activity remains important away from the last line.",
            "goal_threat": "A false nine can score, but elite box output is not the defining target.",
        },
    ),
    "mobile-channel-forward": _prototype(
        "Mobile / Channel Forward",
        "Forwards",
        ("ST",),
        "Attacks space, carries forward and stretches the defence beyond central finishing zones.",
        {
            "movement": 94,
            "dribbling": 88,
            "progression": 84,
            "goal_threat": 80,
            "finishing": 70,
        },
        {
            "movement": "High receiving and running involvement defines the role.",
            "dribbling": "Direct carrying reflects self-propelled progression.",
            "progression": "Advancing the ball and entering dangerous areas captures channel work.",
            "goal_threat": "Runs must still produce attacking danger.",
            "finishing": "Useful scoring output is expected without demanding poacher-level conversion.",
        },
    ),
    "inside-forward": _prototype(
        "Inside Forward",
        "Wide Attackers",
        ("W",),
        "Starts wide but attacks central scoring zones with carries and off-ball movement.",
        {
            "goal_threat": 92,
            "finishing": 86,
            "dribbling": 84,
            "movement": 84,
            "creativity": 62,
        },
        {
            "goal_threat": "Central scoring involvement is the defining destination.",
            "finishing": "Finishing separates the profile from a conventional winger.",
            "dribbling": "Carries enable inward progression.",
            "movement": "Receiving and box occupation capture the move inside.",
            "creativity": "Creation matters, but it is not the primary identity.",
        },
    ),
    "touchline-winger": _prototype(
        "Touchline Winger",
        "Wide Attackers",
        ("W",),
        "Holds width, progresses possession and supplies chances from wide areas.",
        {
            "dribbling": 90,
            "progression": 88,
            "creativity": 84,
            "movement": 80,
            "goal_threat": 48,
        },
        {
            "dribbling": "Beating opponents sustains width and penetration.",
            "progression": "Carries and passes move play up the flank.",
            "creativity": "Chance supply is a core outcome.",
            "movement": "High receiving involvement supports touchline availability.",
            "goal_threat": "The target deliberately does not require inside-forward scoring volume.",
        },
    ),
    "creative-winger": _prototype(
        "Creative Winger",
        "Wide Attackers",
        ("W",),
        "Creates high-value opportunities from wide or half-space possession.",
        {
            "creativity": 96,
            "dribbling": 86,
            "progression": 82,
            "distribution": 76,
            "goal_threat": 56,
        },
        {
            "creativity": "Expected assists, key passes and shot creation define the prototype.",
            "dribbling": "One-versus-one ability opens passing lanes.",
            "progression": "Advancement supports creation in dangerous areas.",
            "distribution": "Passing involvement distinguishes a connector from a pure runner.",
            "goal_threat": "Moderate scoring threat keeps the profile creative-first.",
        },
    ),
    "direct-goal-threat-winger": _prototype(
        "Direct Goal-Threat Winger",
        "Wide Attackers",
        ("W",),
        "Carries aggressively toward goal and generates shots from wide starting positions.",
        {
            "dribbling": 94,
            "progression": 90,
            "goal_threat": 88,
            "movement": 82,
            "creativity": 56,
        },
        {
            "dribbling": "Direct take-ons are the main route forward.",
            "progression": "Territory gain is expected from those actions.",
            "goal_threat": "Shot and box involvement distinguish danger from sterile carrying.",
            "movement": "Receiving volume supports repeated attacks.",
            "creativity": "Creation is useful but intentionally secondary.",
        },
    ),
    "advanced-playmaker": _prototype(
        "Advanced Playmaker",
        "Midfielders",
        ("AM", "CM"),
        "Creates between the lines through passing, receiving and controlled progression.",
        {
            "creativity": 96,
            "distribution": 82,
            "progression": 84,
            "movement": 82,
            "ball_security": 76,
            "goal_threat": 58,
        },
        {
            "creativity": "Chance creation is the central signal.",
            "distribution": "Passing involvement supports orchestration.",
            "progression": "Line-breaking actions move attacks forward.",
            "movement": "Receiving in advanced areas keeps the player available.",
            "ball_security": "Retention supports high-touch creative responsibility.",
            "goal_threat": "Some shooting presence is useful without defining the role.",
        },
    ),
    "deep-lying-playmaker": _prototype(
        "Deep-Lying Playmaker",
        "Midfielders",
        ("DM", "CM"),
        "Controls possession from deeper zones with distribution, progression and security.",
        {
            "distribution": 96,
            "progression": 90,
            "ball_security": 88,
            "creativity": 68,
            "defensive_activity": 54,
        },
        {
            "distribution": "High-volume, accurate circulation is the foundation.",
            "progression": "Forward passing distinguishes control from recycling.",
            "ball_security": "Low-error retention is essential in deep areas.",
            "creativity": "Some chance creation is expected but not at number-ten level.",
            "defensive_activity": "Defensive work is useful without defining a playmaker.",
        },
    ),
    "box-to-box-midfielder": _prototype(
        "Box-to-Box Midfielder",
        "Midfielders",
        ("CM",),
        "Contributes across phases through progression, movement, defending and attacking support.",
        {
            "progression": 88,
            "movement": 90,
            "defensive_activity": 82,
            "distribution": 76,
            "goal_threat": 72,
            "dribbling": 68,
        },
        {
            "progression": "Territory gain links both halves.",
            "movement": "High involvement represents repeated box-to-box availability.",
            "defensive_activity": "Ball recovery is part of the two-way profile.",
            "distribution": "Reliable circulation supports transition between phases.",
            "goal_threat": "Late attacking contribution is expected.",
            "dribbling": "Carrying offers an additional route through midfield.",
        },
    ),
    "ball-winning-midfielder": _prototype(
        "Ball-Winning Midfielder",
        "Midfielders",
        ("DM", "CM"),
        "Wins possession frequently and engages opponents with assertive defensive actions.",
        {
            "defensive_activity": 96,
            "defensive_aggression": 92,
            "aerial": 70,
            "ball_security": 62,
            "progression": 48,
        },
        {
            "defensive_activity": "Tackles, interceptions, blocks and recoveries define the profile.",
            "defensive_aggression": "High engagement separates a ball-winner from a positional holder.",
            "aerial": "Physical duel contribution adds defensive range.",
            "ball_security": "Moderate retention prevents the prototype becoming purely destructive.",
            "progression": "Progression is deliberately not required at playmaker levels.",
        },
    ),
    "progressive-carrier": _prototype(
        "Progressive Carrier",
        "Midfielders",
        ("CM", "AM"),
        "Breaks lines primarily through carries and dribbling rather than static circulation.",
        {
            "dribbling": 95,
            "progression": 94,
            "movement": 86,
            "ball_security": 72,
            "creativity": 66,
        },
        {
            "dribbling": "Take-ons and carry efficiency define the method.",
            "progression": "Territory gain is the intended result.",
            "movement": "Receiving involvement creates opportunities to carry.",
            "ball_security": "Adequate retention balances an inherently risky style.",
            "creativity": "Creation may follow progression but is not required to be elite.",
        },
    ),
    "attacking-wingback": _prototype(
        "Attacking Wingback",
        "Fullbacks / Wingbacks",
        ("FB_WB",),
        "Provides high and wide progression, creation and repeated attacking involvement.",
        {
            "progression": 94,
            "creativity": 90,
            "movement": 92,
            "dribbling": 82,
            "defensive_activity": 58,
        },
        {
            "progression": "Forward carries and passes drive territorial impact.",
            "creativity": "Final-third delivery distinguishes an attacking wingback.",
            "movement": "High receiving and touch involvement reflects repeated overlaps.",
            "dribbling": "Wide one-versus-one ability supports penetration.",
            "defensive_activity": "Some defending remains expected without defining the style.",
        },
    ),
    "overlapping-fullback": _prototype(
        "Overlapping Fullback",
        "Fullbacks / Wingbacks",
        ("FB_WB",),
        "Advances outside the winger through running volume, carries and chance supply.",
        {
            "movement": 95,
            "progression": 92,
            "dribbling": 84,
            "creativity": 78,
            "defensive_activity": 62,
        },
        {
            "movement": "Repeated forward availability is the key signal.",
            "progression": "Carries and final-third entries capture the overlap.",
            "dribbling": "Wide advancement often requires carrying past pressure.",
            "creativity": "Delivery is expected, though below an attacking-wingback prototype.",
            "defensive_activity": "The fullback remains responsible in defensive phases.",
        },
    ),
    "inverted-fullback": _prototype(
        "Inverted Fullback",
        "Fullbacks / Wingbacks",
        ("FB_WB",),
        "Moves into central build-up areas and resembles a secure progressive midfielder in possession.",
        {
            "distribution": 92,
            "ball_security": 90,
            "progression": 86,
            "defensive_activity": 72,
            "creativity": 64,
        },
        {
            "distribution": "Central circulation is the defining possession signal.",
            "ball_security": "Secure handling is critical in central zones.",
            "progression": "Forward passing and carrying retain attacking value.",
            "defensive_activity": "Defensive contribution anchors the hybrid role.",
            "creativity": "Some creation is useful but not required at playmaker levels.",
        },
    ),
    "defensive-fullback": _prototype(
        "Defensive Fullback",
        "Fullbacks / Wingbacks",
        ("FB_WB",),
        "Prioritises defensive interventions, secure possession and restrained progression.",
        {
            "defensive_activity": 94,
            "defensive_aggression": 86,
            "ball_security": 82,
            "aerial": 68,
            "progression": 46,
        },
        {
            "defensive_activity": "Interventions and recoveries define the profile.",
            "defensive_aggression": "Engagement volume captures proactive defending.",
            "ball_security": "Safe retention supports a conservative role.",
            "aerial": "Physical defending adds back-post and duel value.",
            "progression": "The prototype intentionally does not demand attacking-fullback output.",
        },
    ),
    "ball-playing-centre-back": _prototype(
        "Ball-Playing Centre Back",
        "Centre Backs",
        ("CB",),
        "Breaks lines from defence while retaining the ball and maintaining defensive involvement.",
        {
            "distribution": 94,
            "progression": 92,
            "ball_security": 88,
            "defensive_activity": 68,
            "aerial": 62,
        },
        {
            "distribution": "Passing volume and accuracy establish build-up responsibility.",
            "progression": "Forward passing distinguishes line-breaking defenders.",
            "ball_security": "Low-error possession protects central build-up.",
            "defensive_activity": "The player must still operate as a defender.",
            "aerial": "Moderate aerial presence preserves positional plausibility.",
        },
    ),
    "stopper-centre-back": _prototype(
        "Stopper",
        "Centre Backs",
        ("CB",),
        "Steps forward to duel, intercept and disrupt attacks aggressively.",
        {
            "defensive_activity": 95,
            "defensive_aggression": 94,
            "aerial": 82,
            "ball_security": 58,
            "distribution": 44,
        },
        {
            "defensive_activity": "High intervention volume is fundamental.",
            "defensive_aggression": "Proactive engagement separates a stopper from a cover defender.",
            "aerial": "Physical duel strength supports front-foot defending.",
            "ball_security": "Moderate security prevents needless possession risk.",
            "distribution": "Elite build-up contribution is deliberately not required.",
        },
    ),
    "cover-defender": _prototype(
        "Cover Defender",
        "Centre Backs",
        ("CB",),
        "Protects depth through composed positioning, recovery work and secure possession.",
        {
            "defensive_activity": 84,
            "ball_security": 90,
            "progression": 72,
            "aerial": 70,
            "defensive_aggression": 48,
        },
        {
            "defensive_activity": "Reliable defensive output remains necessary.",
            "ball_security": "Composure distinguishes cover from confrontation.",
            "progression": "Useful advancement supports transitions after recovery.",
            "aerial": "Balanced physical coverage remains important.",
            "defensive_aggression": "The lower target reflects restraint rather than weakness.",
        },
    ),
    "aerial-physical-centre-back": _prototype(
        "Aerial / Physical Centre Back",
        "Centre Backs",
        ("CB",),
        "Dominates aerial contests and produces high defensive-action volume.",
        {
            "aerial": 97,
            "defensive_activity": 92,
            "defensive_aggression": 84,
            "ball_security": 58,
            "progression": 42,
        },
        {
            "aerial": "Aerial volume and win rate are the strongest defining characteristics.",
            "defensive_activity": "Clearances, blocks and recoveries reinforce box defence.",
            "defensive_aggression": "Physical engagement supports duel dominance.",
            "ball_security": "Moderate retention keeps the vector positionally credible.",
            "progression": "Progressive build-up is intentionally secondary.",
        },
    ),
    "shot-stopper": _prototype(
        "Shot-Stopper",
        "Goalkeepers",
        ("GK",),
        "Emphasises saves and goals prevented over aggressive sweeping or expansive distribution.",
        {
            "shot_stopping": 96,
            "gk_sweeping": 52,
            "gk_distribution": 54,
            "ball_security": 72,
        },
        {
            "shot_stopping": "Save rate and goals prevented define the prototype.",
            "gk_sweeping": "Average sweeping avoids conflating styles.",
            "gk_distribution": "Distribution is not required to be exceptional.",
            "ball_security": "Secure handling and passing remain desirable.",
        },
    ),
    "sweeper-keeper": _prototype(
        "Sweeper Keeper",
        "Goalkeepers",
        ("GK",),
        "Defends space outside the box while retaining credible shot-stopping and distribution.",
        {
            "gk_sweeping": 96,
            "shot_stopping": 78,
            "gk_distribution": 76,
            "ball_security": 72,
        },
        {
            "gk_sweeping": "Outside-box actions and starting distance define the style.",
            "shot_stopping": "The goalkeeper still needs a strong conventional base.",
            "gk_distribution": "Passing supports the advanced starting position.",
            "ball_security": "Secure possession limits risk behind a high line.",
        },
    ),
    "ball-playing-goalkeeper": _prototype(
        "Ball-Playing Goalkeeper",
        "Goalkeepers",
        ("GK",),
        "Acts as a possession outlet through secure, involved distribution.",
        {
            "gk_distribution": 96,
            "ball_security": 90,
            "gk_sweeping": 76,
            "shot_stopping": 72,
        },
        {
            "gk_distribution": "Passing involvement and completion define the style.",
            "ball_security": "Secure choices support build-up under pressure.",
            "gk_sweeping": "An advanced support position complements possession play.",
            "shot_stopping": "Conventional performance remains relevant but is not the defining axis.",
        },
    ),
}

# Three real FBref-derived fields shown in the existing Scout result-row slots.
# These affect presentation only; cosine uses the dimension subset above.
ARCHETYPE_DISPLAY_METRICS = {
    "poacher-box-striker": (
        "npxg_p90",
        "shots_p90",
        "attacking_penalty_area_touches_p90",
    ),
    "target-forward": ("aerials_won_p90", "aerial_win_pct", "passes_received_p90"),
    "complete-forward": ("npxg_p90", "xag_p90", "successful_takeons_p90"),
    "false-nine": ("xag_p90", "key_passes_p90", "progressive_passes_p90"),
    "mobile-channel-forward": (
        "progressive_receptions_p90",
        "progressive_carries_p90",
        "npxg_p90",
    ),
    "inside-forward": (
        "npxg_p90",
        "successful_takeons_p90",
        "attacking_penalty_area_touches_p90",
    ),
    "touchline-winger": ("successful_takeons_p90", "crosses_p90", "key_passes_p90"),
    "creative-winger": ("xag_p90", "key_passes_p90", "shot_creating_actions_p90"),
    "direct-goal-threat-winger": (
        "successful_takeons_p90",
        "progressive_carries_p90",
        "shots_p90",
    ),
    "advanced-playmaker": ("xag_p90", "key_passes_p90", "passes_penalty_area_p90"),
    "deep-lying-playmaker": (
        "progressive_passes_p90",
        "pass_completion_pct",
        "passes_final_third_p90",
    ),
    "box-to-box-midfielder": (
        "progressive_carries_p90",
        "tackles_won_p90",
        "attacking_penalty_area_touches_p90",
    ),
    "ball-winning-midfielder": (
        "tackles_won_p90",
        "interceptions_p90",
        "recoveries_p90",
    ),
    "progressive-carrier": (
        "progressive_carries_p90",
        "successful_takeons_p90",
        "carries_final_third_p90",
    ),
    "attacking-wingback": ("progressive_carries_p90", "xag_p90", "crosses_p90"),
    "overlapping-fullback": (
        "progressive_receptions_p90",
        "carries_final_third_p90",
        "crosses_p90",
    ),
    "inverted-fullback": (
        "progressive_passes_p90",
        "pass_completion_pct",
        "passes_final_third_p90",
    ),
    "defensive-fullback": ("tackles_won_p90", "interceptions_p90", "blocks_p90"),
    "ball-playing-centre-back": (
        "progressive_passes_p90",
        "pass_completion_pct",
        "progressive_carries_p90",
    ),
    "stopper-centre-back": ("tackles_won_p90", "interceptions_p90", "aerials_won_p90"),
    "cover-defender": ("interceptions_p90", "recoveries_p90", "errors_p90"),
    "aerial-physical-centre-back": (
        "aerials_won_p90",
        "aerial_win_pct",
        "clearances_p90",
    ),
    "shot-stopper": ("goals_prevented_p90", "save_pct", "saves_p90"),
    "sweeper-keeper": (
        "defensive_actions_outside_box_p90",
        "avg_defensive_action_distance",
        "save_pct",
    ),
    "ball-playing-goalkeeper": (
        "gk_pass_completion_pct",
        "gk_passes_attempted_p90",
        "gk_throws_p90",
    ),
}

for _archetype_id, _definition in ARCHETYPE_DEFINITIONS.items():
    _definition["display_metrics"] = ARCHETYPE_DISPLAY_METRICS[_archetype_id]
    _definition["technical_description"] = (
        "Cosine similarity uses only "
        + ", ".join(feature.replace("_", " ") for feature in _definition["features"])
        + "; target coordinates are empirical family quantiles in the shared robust-scaled space."
    )


@dataclass
class ArchetypeArtifacts:
    definitions: dict[str, dict]
    similarities: pd.DataFrame
    raw_cosines: pd.DataFrame
    target_vectors: dict[tuple[str, str], np.ndarray]

    def matches_for_player(self, index: int) -> list[dict]:
        matches = []
        for archetype_id, definition in self.definitions.items():
            score = self.similarities.at[index, archetype_id]
            if pd.isna(score):
                continue
            raw_cosine = round(float(self.raw_cosines.at[index, archetype_id]), 4)
            matches.append(
                {
                    "id": archetype_id,
                    "name": definition["name"],
                    "category": definition["category"],
                    "description": definition["description"],
                    "similarity": round(max(0.0, raw_cosine) * 100.0, 1),
                    "cosine_similarity": raw_cosine,
                }
            )
        return sorted(matches, key=lambda item: item["cosine_similarity"], reverse=True)


def _target_vector(
    values: pd.DataFrame, features: tuple[str, ...], targets: dict[str, float]
) -> np.ndarray:
    """Convert percentile targets into the observed normalized coordinates."""
    return np.array(
        [
            float(
                values[feature].quantile(
                    float(targets[feature]) / 100.0, interpolation="linear"
                )
            )
            for feature in features
        ]
    )


def build_archetype_artifacts(
    players: pd.DataFrame,
    dimensions: pd.DataFrame,
    definitions: dict[str, dict] | None = None,
) -> ArchetypeArtifacts:
    definitions = definitions or ARCHETYPE_DEFINITIONS
    similarities = pd.DataFrame(
        np.nan, index=players.index, columns=list(definitions), dtype=float
    )
    raw_cosines = similarities.copy()
    target_vectors: dict[tuple[str, str], np.ndarray] = {}

    for archetype_id, definition in definitions.items():
        features = tuple(definition["features"])
        missing = [feature for feature in features if feature not in dimensions.columns]
        if missing:
            raise ValueError(
                f"Archetype {archetype_id!r} uses unavailable dimensions: {missing}"
            )
        if tuple(definition["target_percentiles"]) != features:
            raise ValueError(
                f"Archetype {archetype_id!r} target ordering does not match its feature ordering"
            )

        for family in definition["families"]:
            indices = players.index[players["position_family"] == family]
            if len(indices) < 2:
                continue
            family_values = dimensions.loc[indices, list(features)].astype(float)
            target = _target_vector(
                family_values, features, definition["target_percentiles"]
            )
            target_vectors[(archetype_id, family)] = target
            target_norm = np.linalg.norm(target)
            if target_norm == 0:
                continue
            cosines = cosine_similarity(
                family_values.to_numpy(float), target.reshape(1, -1)
            ).ravel()
            raw_cosines.loc[indices, archetype_id] = cosines
            similarities.loc[indices, archetype_id] = np.clip(
                cosines * 100.0, 0.0, 100.0
            )

    return ArchetypeArtifacts(definitions, similarities, raw_cosines, target_vectors)


def archetype_to_api(archetype_id: str, definition: dict) -> dict:
    return {
        "id": archetype_id,
        "name": definition["name"],
        "category": definition["category"],
        "families": list(definition["families"]),
        "description": definition["description"],
        "technical_description": definition["technical_description"],
        "display_metrics": list(definition["display_metrics"]),
        "features": [
            {
                "id": feature,
                "label": feature.replace("_", " ").title(),
                "target_percentile": float(definition["target_percentiles"][feature]),
                "rationale": definition["rationale"][feature],
            }
            for feature in definition["features"]
        ],
        "method": "Cosine similarity between the player's robust-scaled dimension vector and a human-defined prototype vector in the same position-family feature space.",
        "display_scale": "raw cosine × 100, clamped to 0–100; this is similarity, not performance or probability",
    }
