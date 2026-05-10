"""
Chargement des données pour le TP1 - Recommandation item-item.
Supporte MovieLens 100K (téléchargeable) et un jeu synthétique de démo.
"""

import io
import os
import zipfile
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import requests


# ──────────────────────────────────────────────────────────────
#  Données synthétiques (mode démo, sans téléchargement)
# ──────────────────────────────────────────────────────────────

MOVIES = {
    1:  "The Matrix",
    2:  "Inception",
    3:  "Interstellar",
    4:  "The Dark Knight",
    5:  "Pulp Fiction",
    6:  "Fight Club",
    7:  "Forrest Gump",
    8:  "The Shawshank Redemption",
    9:  "Goodfellas",
    10: "The Silence of the Lambs",
    11: "Schindler's List",
    12: "The Godfather",
    13: "Jurassic Park",
    14: "Titanic",
    15: "Avatar",
    16: "The Lion King",
    17: "Toy Story",
    18: "Finding Nemo",
    19: "WALL-E",
    20: "Up",
    21: "Coco",
    22: "Soul",
    23: "Parasite",
    24: "Spirited Away",
    25: "Your Name",
}

GENRES = {
    1:  "Sci-Fi/Action",
    2:  "Sci-Fi/Thriller",
    3:  "Sci-Fi/Drama",
    4:  "Action/Crime",
    5:  "Crime/Drama",
    6:  "Drama/Thriller",
    7:  "Drama/Romance",
    8:  "Drama",
    9:  "Crime/Drama",
    10: "Thriller",
    11: "Drama/History",
    12: "Crime/Drama",
    13: "Adventure/Sci-Fi",
    14: "Drama/Romance",
    15: "Sci-Fi/Action",
    16: "Animation",
    17: "Animation",
    18: "Animation",
    19: "Animation/Sci-Fi",
    20: "Animation/Drama",
    21: "Animation",
    22: "Animation/Drama",
    23: "Drama/Thriller",
    24: "Animation/Fantasy",
    25: "Animation/Romance",
}


def generate_synthetic_data(
    n_users: int = 50,
    n_items: int = 25,
    density: float = 0.45,
    seed: int = 42,
) -> Tuple[pd.DataFrame, Dict, Dict]:
    """
    Génère un dataset synthétique de notes (1-5).

    Simule des profils d'utilisateurs avec des affinités par genre
    pour rendre les patterns de similarité item-item significatifs.

    Returns:
        ratings_df : DataFrame ['userId', 'itemId', 'rating']
        movies     : dict {itemId: title}
        genres     : dict {itemId: genre}
    """
    rng = np.random.default_rng(seed)

    # Profils utilisateurs : affinité animation vs cinéma réaliste
    animation_items = [16, 17, 18, 19, 20, 21, 22, 24, 25]
    realistic_items = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

    rows = []
    for user_id in range(1, n_users + 1):
        # Chaque user note ~density*n_items films
        n_rated = max(3, int(rng.binomial(n_items, density)))
        rated_items = rng.choice(range(1, n_items + 1), size=n_rated, replace=False)

        # Biais personnel
        personal_bias = rng.normal(0, 0.4)
        # Préférence animation
        anim_fan = rng.random() > 0.5

        for item_id in rated_items:
            base = 3.0 + personal_bias
            # Bonus genre
            if anim_fan and item_id in animation_items:
                base += rng.uniform(0.5, 1.5)
            elif not anim_fan and item_id in realistic_items:
                base += rng.uniform(0.5, 1.5)
            # Bruit
            rating = base + rng.normal(0, 0.6)
            rating = float(np.clip(round(rating * 2) / 2, 1.0, 5.0))
            rows.append({"userId": user_id, "itemId": int(item_id), "rating": rating})

    ratings_df = pd.DataFrame(rows)
    return ratings_df, MOVIES, GENRES


# ──────────────────────────────────────────────────────────────
#  MovieLens 100K (optionnel)
# ──────────────────────────────────────────────────────────────

ML100K_URL = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
ML100K_CACHE = Path("data/ml-100k")


def load_movielens_100k() -> Optional[Tuple[pd.DataFrame, Dict, Dict]]:
    """
    Charge MovieLens 100K.
    Télécharge et cache si nécessaire. Retourne None si indisponible.
    """
    ratings_path = ML100K_CACHE / "u.data"
    items_path   = ML100K_CACHE / "u.item"

    # Cache local
    if not ratings_path.exists():
        try:
            ML100K_CACHE.mkdir(parents=True, exist_ok=True)
            resp = requests.get(ML100K_URL, timeout=15)
            resp.raise_for_status()
            with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
                for name in z.namelist():
                    fname = Path(name).name
                    if fname in ("u.data", "u.item"):
                        with z.open(name) as src, open(ML100K_CACHE / fname, "wb") as dst:
                            dst.write(src.read())
        except Exception:
            return None

    # Chargement ratings
    ratings_df = pd.read_csv(
        ratings_path,
        sep="\t",
        names=["userId", "itemId", "rating", "timestamp"]
    )[["userId", "itemId", "rating"]]

    # Chargement titres films (encodage latin-1)
    movies = {}
    genres_map = {}
    genre_cols = [
        "unknown","Action","Adventure","Animation","Children's","Comedy",
        "Crime","Documentary","Drama","Fantasy","Film-Noir","Horror",
        "Musical","Mystery","Romance","Sci-Fi","Thriller","War","Western"
    ]
    with open(items_path, encoding="latin-1") as f:
        for line in f:
            parts = line.strip().split("|")
            if len(parts) < 24:
                continue
            movie_id = int(parts[0])
            title    = parts[1]
            genres_present = [g for g, v in zip(genre_cols, parts[5:]) if v == "1"]
            movies[movie_id] = title
            genres_map[movie_id] = "/".join(genres_present[:2]) if genres_present else "Unknown"

    return ratings_df, movies, genres_map


def load_data(source: str = "synthetic", **kwargs):
    """
    Point d'entrée unique.

    Args:
        source: 'synthetic' | 'movielens'
    """
    if source == "movielens":
        result = load_movielens_100k()
        if result is not None:
            return result
        # Fallback
    return generate_synthetic_data(**kwargs)
