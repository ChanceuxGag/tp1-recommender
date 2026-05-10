"""
Moteur de recommandation : Collaborative Filtering Item-Item (Top-N)
TP1 - Systèmes de recommandation
"""

from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


class ItemItemCF:
    """
    Système de recommandation par filtrage collaboratif item-item.

    Algorithme :
    1. Construire la matrice User × Item (ratings)
    2. Calculer la similarité cosinus entre chaque paire d'items
    3. Pour un utilisateur donné, prédire les ratings des items non notés
       via la moyenne pondérée des ratings sur les items similaires
    4. Retourner les Top-N items avec le score prédit le plus élevé
    """

    def __init__(self, min_common_users: int = 2):
        """
        Args:
            min_common_users: Nombre minimum d'utilisateurs communs pour
                              considérer la similarité entre deux items.
        """
        self.min_common_users = min_common_users
        self.user_item_matrix: Optional[pd.DataFrame] = None
        self.similarity_matrix: Optional[pd.DataFrame] = None
        self.item_ids: List = []
        self.user_ids: List = []

    # ------------------------------------------------------------------
    # 1. Ajustement (fit)
    # ------------------------------------------------------------------
    def fit(self, ratings_df: pd.DataFrame) -> "ItemItemCF":
        """
        Entraîne le modèle à partir d'un DataFrame de notes.

        Args:
            ratings_df: DataFrame avec colonnes ['userId', 'itemId', 'rating']

        Returns:
            self (pour le chaining)
        """
        # ① Construire la matrice User × Item
        self.user_item_matrix = ratings_df.pivot_table(
            index="userId",
            columns="itemId",
            values="rating",
            aggfunc="mean"
        )

        self.user_ids = list(self.user_item_matrix.index)
        self.item_ids = list(self.user_item_matrix.columns)

        # ② Calculer la matrice de similarité Item × Item
        # On remplace les NaN par 0 pour le calcul cosinus
        matrix_filled = self.user_item_matrix.fillna(0).values.T  # shape: (n_items, n_users)
        sim_values = cosine_similarity(matrix_filled)

        # Mettre NaN sur la diagonale (un item n'est pas similaire à lui-même)
        np.fill_diagonal(sim_values, np.nan)

        self.similarity_matrix = pd.DataFrame(
            sim_values,
            index=self.item_ids,
            columns=self.item_ids
        )

        return self

    # ------------------------------------------------------------------
    # 2. Prédiction d'un rating
    # ------------------------------------------------------------------
    def predict_rating(self, user_id, item_id, k: int = 10) -> float:
        """
        Prédit le rating qu'un utilisateur donnerait à un item.

        Formule :
            r̂(u,i) = Σ [ sim(i,j) × r(u,j) ] / Σ |sim(i,j)|
            où la somme porte sur les k items les plus similaires à i
            que l'utilisateur u a déjà notés.

        Args:
            user_id: identifiant de l'utilisateur
            item_id: identifiant de l'item cible
            k: nombre de voisins (items similaires) à considérer

        Returns:
            rating prédit (float), ou NaN si impossible
        """
        if user_id not in self.user_ids or item_id not in self.item_ids:
            return np.nan

        # Notes de l'utilisateur sur tous les autres items
        user_ratings = self.user_item_matrix.loc[user_id].dropna()
        # Exclure l'item cible lui-même
        user_rated_items = user_ratings.index.difference([item_id])

        if len(user_rated_items) == 0:
            return np.nan

        # Similarités entre l'item cible et les items notés par l'utilisateur
        similarities = self.similarity_matrix.loc[item_id, user_rated_items].dropna()

        if len(similarities) == 0:
            return np.nan

        # Garder les k voisins les plus similaires (positifs uniquement)
        top_k_sims = similarities.nlargest(k)
        top_k_sims = top_k_sims[top_k_sims > 0]

        if len(top_k_sims) == 0:
            return np.nan

        # Moyenne pondérée
        ratings_for_neighbors = user_ratings[top_k_sims.index]
        numerator = (top_k_sims * ratings_for_neighbors).sum()
        denominator = top_k_sims.abs().sum()

        return numerator / denominator if denominator != 0 else np.nan

    # ------------------------------------------------------------------
    # 3. Top-N recommandations pour un utilisateur
    # ------------------------------------------------------------------
    def recommend(self, user_id, n: int = 10, k_neighbors: int = 10) -> pd.DataFrame:
        """
        Génère les Top-N recommandations pour un utilisateur.

        Args:
            user_id: identifiant de l'utilisateur
            n: nombre de recommandations à retourner
            k_neighbors: nombre de voisins pour la prédiction

        Returns:
            DataFrame ['itemId', 'predicted_rating', 'n_neighbors']
        """
        if user_id not in self.user_ids:
            raise ValueError(f"Utilisateur '{user_id}' introuvable.")

        # Items déjà notés par l'utilisateur
        already_rated = set(
            self.user_item_matrix.loc[user_id].dropna().index
        )
        # Items non encore notés
        candidate_items = [i for i in self.item_ids if i not in already_rated]

        results = []
        for item_id in candidate_items:
            pred = self.predict_rating(user_id, item_id, k=k_neighbors)
            if not np.isnan(pred):
                # Compter le nb de voisins effectivement utilisés
                user_ratings = self.user_item_matrix.loc[user_id].dropna()
                sims = self.similarity_matrix.loc[item_id, user_ratings.index].dropna()
                sims_pos = sims[sims > 0].nlargest(k_neighbors)
                results.append({
                    "itemId": item_id,
                    "predicted_rating": round(pred, 3),
                    "n_neighbors": len(sims_pos)
                })

        if not results:
            return pd.DataFrame(columns=["itemId", "predicted_rating", "n_neighbors"])

        recs = pd.DataFrame(results).sort_values("predicted_rating", ascending=False)
        return recs.head(n).reset_index(drop=True)

    # ------------------------------------------------------------------
    # 4. Helpers / statistiques
    # ------------------------------------------------------------------
    def get_user_history(self, user_id) -> pd.Series:
        """Retourne les items notés par l'utilisateur, triés par note."""
        if user_id not in self.user_ids:
            raise ValueError(f"Utilisateur '{user_id}' introuvable.")
        return self.user_item_matrix.loc[user_id].dropna().sort_values(ascending=False)

    def get_item_similarities(self, item_id, top_k: int = 10) -> pd.Series:
        """Retourne les k items les plus similaires à un item donné."""
        if item_id not in self.item_ids:
            raise ValueError(f"Item '{item_id}' introuvable.")
        return self.similarity_matrix[item_id].dropna().nlargest(top_k)

    def sparsity(self) -> float:
        """Retourne le taux de sparsité de la matrice."""
        total = self.user_item_matrix.size
        filled = self.user_item_matrix.count().sum()
        return round(1 - filled / total, 4)

    @property
    def n_users(self) -> int:
        return len(self.user_ids)

    @property
    def n_items(self) -> int:
        return len(self.item_ids)
