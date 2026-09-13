"""
similarity.py — Stage 4c: Similarity engine

The core "find players like X" function. Uses cosine similarity by
default (compares playstyle shape, not raw magnitude) over the
weighted, scaled feature vectors. Can restrict to same position
group so a CB search doesn't return strikers.
"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


class PlayerSimilarityEngine:
    def __init__(
        self,
        df: pd.DataFrame,
        feature_matrix: np.ndarray,
        feature_columns: list[str],
        family_weights: dict[str, np.ndarray] | None = None,
    ):
        """
        df: the player metadata dataframe (must be row-aligned with feature_matrix)
        feature_matrix: the scaled/weighted numeric feature matrix
        feature_columns: names of the columns in feature_matrix, in order
        """
        self.df = df.reset_index(drop=True)
        self.matrix = feature_matrix
        self.feature_columns = feature_columns
        self.family_weights = family_weights or {}
        self._name_to_idx = {name.lower(): i for i, name in enumerate(self.df["name"])}

    def _resolve_index(self, player_name: str) -> int:
        key = player_name.lower()
        if key not in self._name_to_idx:
            close = [n for n in self.df["name"] if key in n.lower()]
            hint = f" Did you mean {close[0]}?" if close else ""
            raise ValueError(f"Player '{player_name}' not found.{hint}")
        return self._name_to_idx[key]

    def resolve_index(self, player_name: str) -> int:
        """Return the row index for a player using the engine's name matching."""
        return self._resolve_index(player_name)

    def score_candidates(
        self, player_name: str, same_position_only: bool = True
    ) -> tuple[int, np.ndarray, np.ndarray]:
        """Return target index, candidate indices, and cosine scores.

        Scores are calculated in the weighted, z-scored feature space. The
        returned cosine values are deliberately kept on the [-1, 1] scale so
        callers can choose their display format without changing the metric.
        """
        idx = self._resolve_index(player_name)
        family = (
            self.df.loc[idx, "position_family"] if "position_family" in self.df else ""
        )
        weight_vector = self.family_weights.get(family, np.ones(self.matrix.shape[1]))
        target_vec = (self.matrix[idx] * weight_vector).reshape(1, -1)

        candidate_mask = np.ones(len(self.df), dtype=bool)
        if same_position_only:
            target_group = self.df.loc[idx, "position_group"]
            candidate_mask = (
                (self.df["position_group"] == target_group).to_numpy().copy()
            )
            if "position_families" in self.df and family:
                # Keep hybrid overlap: a W/ST can appear for either family.
                overlap = (
                    self.df["position_families"]
                    .map(lambda values: family in values)
                    .to_numpy()
                )
                if overlap.sum() > 5:
                    candidate_mask &= overlap
        candidate_mask[idx] = False

        candidate_indices = np.where(candidate_mask)[0]
        if len(candidate_indices) == 0:
            return idx, candidate_indices, np.array([], dtype=float)

        scores = cosine_similarity(
            target_vec, self.matrix[candidate_indices] * weight_vector
        )[0]
        return idx, candidate_indices, scores

    def find_similar(
        self, player_name: str, n: int = 5, same_position_only: bool = True
    ) -> pd.DataFrame:
        idx, candidate_indices, sims = self.score_candidates(
            player_name, same_position_only=same_position_only
        )
        if len(candidate_indices) == 0:
            return pd.DataFrame(
                columns=[
                    "name",
                    "position_group",
                    "league",
                    "similarity",
                    "cosine_similarity",
                ]
            )

        order = np.argsort(sims)[::-1][:n]
        top_indices = candidate_indices[order]
        top_sims = sims[order]
        # Display the actual cosine as a percentage.  A cosine of 0.9742 is
        # therefore shown as 97.4%, rather than being remapped to a peer-rank
        # percentile that can misleadingly read 100%.
        displayed_cosines = top_sims.round(4)
        display_scores = np.array(
            [round(max(0.0, float(score)) * 100.0, 1) for score in displayed_cosines]
        )

        result = self.df.loc[top_indices, ["name", "position_group", "league"]].copy()
        result["similarity"] = display_scores
        result["cosine_similarity"] = displayed_cosines
        if "position_family" in self.df:
            result["position_family"] = self.df.loc[
                top_indices, "position_family"
            ].to_numpy()
        return result.reset_index(drop=True)
