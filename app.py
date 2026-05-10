"""
TP1 - Système de recommandation : Collaborative Filtering Item-Item (Top-N)
Application Streamlit

Etudiant Camus gael TOGBE
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from data_loader import load_data
from recommender import ItemItemCF

# ──────────────────────────────────────────────────────────────
#  Configuration de la page
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Camus System · Item-Item CF",
    page_icon="Royal Immo.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    /* Bannière titre */
    .main-title {
        font-size: 2rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        color: #555;
        font-size: 0.95rem;
        margin-bottom: 1.5rem;
    }
    /* Cards métriques */
    .metric-card {
        background: #f8f9fe;
        border: 1px solid #e0e4f5;
        border-radius: 10px;
        padding: 16px 20px;
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #3d52a0;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #777;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    /* Badges genre */
    .badge {
        display: inline-block;
        background: #ede9fe;
        color: #5b21b6;
        border-radius: 20px;
        padding: 2px 10px;
        font-size: 0.78rem;
        margin: 2px;
    }
    /* Barre de score */
    .score-bar-container {
        background: #eee;
        border-radius: 4px;
        height: 8px;
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
#  Chargement & cache du modèle
# ──────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Chargement des données…")
def get_data(source: str, n_users: int = 50, density: float = 0.45):
    return load_data(source=source, n_users=n_users, density=density)


@st.cache_resource(show_spinner="Entraînement du modèle…")
def get_model(ratings_key: str, _ratings_df: pd.DataFrame):
    model = ItemItemCF()
    model.fit(_ratings_df)
    return model


# ──────────────────────────────────────────────────────────────
#  SIDEBAR
# ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("##  Paramètres")

    # Source de données
    source = st.radio(
        "Jeu de données",
        ["Synthétique (démo)", "MovieLens 100K"],
        index=0,
        help="MovieLens 100K sera téléchargé (~5 Mo) si indisponible localement."
    )
    data_source = "synthetic" if "Synthétique" in source else "movielens"

    st.divider()

    # Paramètres synthétiques
    if data_source == "synthetic":
        n_users = st.slider("Nombre d'utilisateurs", 20, 200, 50, 10)
        density = st.slider("Densité de notation", 0.20, 0.80, 0.45, 0.05,
                            help="Fraction d'items notés par utilisateur")
    else:
        n_users, density = 943, 0.06

    st.divider()

    # Paramètres du modèle
    st.markdown("### Modèle Item-Item CF")
    top_n = st.slider("Top-N recommandations", 3, 20, 10)
    k_neighbors = st.slider(
        "k voisins (items similaires)", 3, 30, 10,
        help="Nombre maximum d'items similaires utilisés pour prédire un rating"
    )

    st.divider()
    st.markdown("### Visualisation")
    heatmap_size = st.slider(
        "Items dans la heatmap", 5, 30, 15,
        help="Nombre d'items affichés dans la matrice de similarité"
    )

    st.divider()
    st.caption("TP1 · Recommandation · Collaborative Filtering")


# ──────────────────────────────────────────────────────────────
#  Chargement & entraînement
# ──────────────────────────────────────────────────────────────
ratings_df, movies, genres = get_data(data_source, n_users=n_users, density=density)
# Ajouter les titres
ratings_df["title"] = ratings_df["itemId"].map(movies)

model = get_model(f"{data_source}_{n_users}_{density}", ratings_df)


# ──────────────────────────────────────────────────────────────
#  EN-TÊTE
# ──────────────────────────────────────────────────────────────
st.markdown('<p class="main-title">Système de recommandation — Item-Item CF</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Filtrage collaboratif item-item avec similarité cosinus · Top-N recommandations</p>',
    unsafe_allow_html=True
)

# Métriques globales
c1, c2, c3, c4 = st.columns(4)
sparsity_pct = model.sparsity() * 100

for col, val, label in [
    (c1, model.n_users, "Utilisateurs"),
    (c2, model.n_items, "Items"),
    (c3, ratings_df.shape[0], "Notes totales"),
    (c4, f"{sparsity_pct:.1f}%", "Sparsité"),
]:
    col.markdown(
        f'<div class="metric-card">'
        f'<div class="metric-value">{val}</div>'
        f'<div class="metric-label">{label}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

st.markdown("---")


# ──────────────────────────────────────────────────────────────
#  ONGLETS PRINCIPAUX
# ──────────────────────────────────────────────────────────────
tab_reco, tab_sim, tab_matrix, tab_expl = st.tabs([
    " Recommandations",
    " Similarité items",
    " Matrice & données",
    " Explication"
])


# ═══════════════════════════════════════════════
# ONGLET 1 — Recommandations utilisateur
# ═══════════════════════════════════════════════
with tab_reco:
    col_left, col_right = st.columns([1, 2], gap="large")

    with col_left:
        st.markdown("### Sélection utilisateur")
        user_id = st.selectbox(
            "Utilisateur",
            model.user_ids,
            format_func=lambda u: f"User {u}"
        )

        # Historique
        st.markdown("#### Historique de notation")
        history = model.get_user_history(user_id)
        if history.empty:
            st.info("Cet utilisateur n'a pas encore noté d'items.")
        else:
            hist_df = pd.DataFrame({
                "Film": [movies.get(i, str(i)) for i in history.index],
                "Note ": history.values,
                "Genre": [genres.get(i, "?") for i in history.index],
            })
            # Bar chart historique
            fig_hist = px.bar(
                hist_df,
                x="Note ",
                y="Film",
                orientation="h",
                color="Note ",
                color_continuous_scale="Blues",
                range_color=[1, 5],
                text="Note ",
                height=max(250, len(hist_df) * 28)
            )
            fig_hist.update_traces(textposition="outside")
            fig_hist.update_layout(
                margin=dict(l=10, r=10, t=20, b=10),
                showlegend=False,
                coloraxis_showscale=False,
                yaxis={"categoryorder": "total ascending"},
            )
            st.plotly_chart(fig_hist, use_container_width=True)

    with col_right:
        st.markdown(f"###  Top-{top_n} recommandations pour User {user_id}")

        with st.spinner("Calcul des recommandations…"):
            try:
                recs = model.recommend(user_id, n=top_n, k_neighbors=k_neighbors)
            except ValueError as e:
                st.error(str(e))
                recs = pd.DataFrame()

        if recs.empty:
            st.warning("Pas assez de données pour générer des recommandations.")
        else:
            # Enrichir avec titres & genres
            recs["title"] = recs["itemId"].map(movies)
            recs["genre"] = recs["itemId"].map(genres)

            # Affichage en cartes
            for rank, row in recs.iterrows():
                score = row["predicted_rating"]
                stars = "Super" * int(round(score))
                pct = (score - 1) / 4 * 100

                st.markdown(f"""
                <div style="
                    border: 1px solid #e0e4f5;
                    border-radius: 10px;
                    padding: 12px 16px;
                    margin-bottom: 8px;
                    background: {'#f0f4ff' if rank % 2 == 0 else '#fff'};
                ">
                  <div style="display:flex; justify-content:space-between; align-items:center">
                    <div>
                      <span style="font-size:1.05rem;font-weight:600">#{rank+1} — {row['title']}</span>
                      <span class="badge" style="margin-left:8px">{row['genre']}</span>
                    </div>
                    <div style="font-size:1.1rem;font-weight:700;color:#3d52a0">{score:.2f} {stars}</div>
                  </div>
                  <div class="score-bar-container" style="margin-top:8px">
                    <div style="background:#3d52a0;border-radius:4px;height:8px;width:{pct:.0f}%"></div>
                  </div>
                  <div style="font-size:0.78rem;color:#888;margin-top:4px">
                    Basé sur {row['n_neighbors']} film(s) similaire(s) noté(s)
                  </div>
                </div>
                """, unsafe_allow_html=True)

            # Graphe résumé
            fig_rec = px.bar(
                recs,
                x="predicted_rating",
                y="title",
                orientation="h",
                color="predicted_rating",
                color_continuous_scale="RdYlGn",
                range_color=[1, 5],
                labels={"predicted_rating": "Score prédit", "title": "Film"},
                text=recs["predicted_rating"].apply(lambda x: f"{x:.2f}"),
            )
            fig_rec.update_traces(textposition="outside")
            fig_rec.update_layout(
                margin=dict(l=10, r=10, t=20, b=10),
                coloraxis_showscale=False,
                yaxis={"categoryorder": "total ascending"},
                height=max(300, len(recs) * 32),
            )
            st.plotly_chart(fig_rec, use_container_width=True)


# ═══════════════════════════════════════════════
# ONGLET 2 — Similarité entre items
# ═══════════════════════════════════════════════
with tab_sim:
    st.markdown("###  Explorer la similarité entre items")
    col_a, col_b = st.columns(2)

    with col_a:
        ref_item = st.selectbox(
            "Item de référence",
            model.item_ids,
            format_func=lambda i: movies.get(i, str(i)),
        )

        k_sim = st.slider("Nombre de voisins à afficher", 3, min(20, model.n_items - 1), 8)

    with col_b:
        sims = model.get_item_similarities(ref_item, top_k=k_sim)
        sim_df = pd.DataFrame({
            "Film": [movies.get(i, str(i)) for i in sims.index],
            "Genre": [genres.get(i, "?") for i in sims.index],
            "Similarité cosinus": sims.values.round(4),
        })

        st.markdown(f"**Films les plus similaires à : _{movies.get(ref_item, ref_item)}_**")
        if sim_df.empty:
            st.info("Pas de similarité calculable.")
        else:
            st.dataframe(
                sim_df.style.background_gradient(cmap="Blues", subset=["Similarité cosinus"]),
                use_container_width=True,
                hide_index=True,
            )

    # Graphe barres similarités
    if not sim_df.empty:
        fig_sim = px.bar(
            sim_df,
            y="Film",
            x="Similarité cosinus",
            orientation="h",
            color="Similarité cosinus",
            color_continuous_scale="Blues",
            range_color=[0, 1],
            text="Similarité cosinus",
        )
        fig_sim.update_traces(texttemplate="%{text:.3f}", textposition="outside")
        fig_sim.update_layout(
            yaxis={"categoryorder": "total ascending"},
            coloraxis_showscale=False,
            margin=dict(l=10, r=60, t=20, b=10),
            height=max(250, len(sim_df) * 35),
        )
        st.plotly_chart(fig_sim, use_container_width=True)


# ═══════════════════════════════════════════════
# ONGLET 3 — Matrice & données brutes
# ═══════════════════════════════════════════════
with tab_matrix:
    st.markdown("###  Matrice de similarité item-item")

    # Sous-ensemble d'items
    top_items = (
        ratings_df.groupby("itemId")["rating"].count()
        .nlargest(heatmap_size).index.tolist()
    )
    sim_sub = model.similarity_matrix.loc[top_items, top_items]
    item_labels = [movies.get(i, str(i)) for i in top_items]

    fig_heat = go.Figure(data=go.Heatmap(
        z=sim_sub.values,
        x=item_labels,
        y=item_labels,
        colorscale="RdBu",
        zmid=0,
        zmin=-1, zmax=1,
        colorbar=dict(title="Similarité"),
        text=np.round(sim_sub.values, 2),
        texttemplate="%{text}",
        textfont={"size": 9},
        hoverongaps=False,
    ))
    fig_heat.update_layout(
        margin=dict(l=20, r=20, t=30, b=20),
        xaxis=dict(tickangle=-40, tickfont=dict(size=10)),
        yaxis=dict(tickfont=dict(size=10)),
        height=550,
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    st.divider()

    st.markdown("###  Données brutes")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Échantillon du dataset**")
        sample = ratings_df.head(200)[["userId", "title", "rating"]].rename(
            columns={"title": "Film"}
        )
        st.dataframe(sample, use_container_width=True, hide_index=True)

    with col2:
        st.markdown("**Distribution des notes**")
        fig_dist = px.histogram(
            ratings_df,
            x="rating",
            nbins=9,
            color_discrete_sequence=["#3d52a0"],
            labels={"rating": "Note", "count": "Fréquence"},
        )
        fig_dist.update_layout(
            margin=dict(l=10, r=10, t=20, b=10),
            bargap=0.1,
        )
        st.plotly_chart(fig_dist, use_container_width=True)

        st.markdown("**Films les plus notés**")
        pop = (
            ratings_df.groupby("title")["rating"]
            .agg(["count", "mean"])
            .rename(columns={"count": "# notes", "mean": "Moy."})
            .sort_values("# notes", ascending=False)
            .head(10)
            .round(2)
        )
        st.dataframe(pop, use_container_width=True)


# ═══════════════════════════════════════════════
# ONGLET 4 — Explication 
# ═══════════════════════════════════════════════
with tab_expl:
    st.markdown("##  Comment fonctionne l'algorithme ?")

    st.markdown("""
    ### 1. Construction de la matrice User × Item

    On organise les notes dans une matrice **R** de dimensions `(n_users × n_items)`.
    Chaque cellule `R[u, i]` contient la note donnée par l'utilisateur **u** à l'item **i**.
    Les cellules non renseignées sont des **NaN** (sparsité).

    ```
              Film A   Film B   Film C   Film D
    User 1     4.5      3.0      NaN      5.0
    User 2     NaN      4.0      2.0      NaN
    User 3     3.5      NaN      4.0      4.5
    ```
    """)

    st.markdown("""
    ### 2. Similarité cosinus entre items

    Pour mesurer à quel point deux items **i** et **j** sont similaires,
    on considère leurs vecteurs de notes (sur les utilisateurs communs) et
    on calcule le **cosinus de l'angle** entre eux :

    $$
    \\text{sim}(i, j) = \\frac{\\mathbf{r}_i \\cdot \\mathbf{r}_j}{\\|\\mathbf{r}_i\\| \\cdot \\|\\mathbf{r}_j\\|}
    = \\frac{\\sum_u r_{u,i} \\cdot r_{u,j}}{\\sqrt{\\sum_u r_{u,i}^2} \\cdot \\sqrt{\\sum_u r_{u,j}^2}}
    $$

    - Résultat ∈ [−1, 1] : **1** = identiques, **0** = orthogonaux, **−1** = opposés.
    """)

    st.markdown("""
    ### 3. Prédiction du rating

    Pour prédire la note que l'utilisateur **u** donnerait à l'item **i** :

    1. On trouve les **k items les plus similaires à i** que **u a déjà notés** → voisins N(i, u)
    2. On calcule la **moyenne pondérée** par les similarités :

    $$
    \\hat{r}_{u,i} = \\frac{\\sum_{j \\in N(i,u)} \\text{sim}(i,j) \\cdot r_{u,j}}{\\sum_{j \\in N(i,u)} |\\text{sim}(i,j)|}
    $$
    """)

    st.markdown("""
    ### 4. Top-N recommandations

    - On calcule `r̂(u, i)` pour **tous les items non encore notés** par l'utilisateur.
    - On trie par score prédit décroissant.
    - On retourne les **N premiers** items.
    """)

    st.info("""
    **Avantages** de l'approche item-item vs user-user :
    - La matrice de similarité items est **stable** (les items ne changent pas).
    - Meilleure **scalabilité** : O(n_items²) à précalculer une fois.
    - Fonctionne bien quand n_users >> n_items.
    """)

    st.markdown("""
    ### Paramètres clés de ce TP

    | Paramètre | Rôle |
    |-----------|------|
    | **k voisins** | Nombre d'items similaires utilisés pour prédire (biais/variance tradeoff) |
    | **Top-N** | Taille de la liste de recommandations finale |
    | **Densité** | Taux de remplissage de la matrice (impact sur la qualité) |
    | **Sparsité** | 1 − densité : problème classique des systèmes de reco |
    """)
