# TP1 — Système de recommandation : Item-Item Collaborative Filtering

## Installation

```bash
pip install -r requirements.txt
```

## Lancement

```bash
streamlit run app.py
```

L'application s'ouvre automatiquement sur http://localhost:8501

---

## Structure du projet

```
├── app.py           # Interface Streamlit
├── recommender.py   # Algorithme Item-Item CF (à étudier !)
├── data_loader.py   # Chargement données synthétiques / MovieLens 100K
├── requirements.txt
└── README.md
```

---

## Algorithme Item-Item CF (résumé)

### Étape 1 — Matrice User × Item
```python
R = ratings_df.pivot_table(index='userId', columns='itemId', values='rating')
```

### Étape 2 — Similarité cosinus
```python
from sklearn.metrics.pairwise import cosine_similarity
S = cosine_similarity(R.fillna(0).T)  # shape: (n_items, n_items)
```

### Étape 3 — Prédiction
```
r̂(u,i) = Σ [sim(i,j) × r(u,j)] / Σ |sim(i,j)|
          où j ∈ top-k items similaires à i notés par u
```

### Étape 4 — Top-N
```python
recs = predictions.nlargest(N)
```

---

## Paramètres explorables dans l'interface

| Paramètre | Effet |
|-----------|-------|
| **Top-N** | Taille de la liste de recommandations |
| **k voisins** | Nombre d'items similaires utilisés pour prédire |
| **Densité** | Taux de remplissage de la matrice utilisateur-item |
| **Jeu de données** | Synthétique (instantané) ou MovieLens 100K (~5 Mo) |

---
