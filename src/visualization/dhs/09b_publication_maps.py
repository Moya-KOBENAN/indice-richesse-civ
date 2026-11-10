"""Étape historique isolée : exécution explicite uniquement via main()."""


def main():
    # Résolution des chemins indépendante du répertoire de lancement.
    from pathlib import Path
    import sys

    _PROJECT_ROOT = Path(__file__).resolve().parents[3]
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))

    from src.utils.paths import (
        DHS_MAPS_DIR,
        DHS_TABLES_DIR,
        PROJECT_ROOT,
        RAW_DIR,
        ensure_layout,
    )

    ensure_layout('dhs')


    import pandas as pd
    import numpy as np
    import geopandas as gpd
    import matplotlib.pyplot as plt


    # ============================================================
    # 09B - CARTES AMELIORÉES
    # ============================================================


    PREDICTIONS_FILE = (
        DHS_TABLES_DIR / 'spatial_block_robustness_predictions.csv'
    )

    OUTPUT_DIR = (
        DHS_MAPS_DIR / 'publication'
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    BOUNDARY_FILE = (
        RAW_DIR / 'cote_ivoire_boundary.geojson'
    )


    # ============================================================
    # 1. CHARGEMENT DES PREDICTIONS
    # ============================================================

    print("=" * 72)
    print("ETAPE 09B - CARTES AMELIORÉES")
    print("=" * 72)

    df = pd.read_csv(PREDICTIONS_FILE)

    print(f"Clusters chargés : {len(df)}")

    if len(df) != 539:
        raise ValueError(
            f"539 clusters attendus, {len(df)} obtenus."
        )


    required = [
        "cluster",
        "longitude",
        "latitude",
        "wealth_observed",
        "pred_viirs",
        "pred_cnn",
        "pred_cnn_plus_viirs",
    ]

    for col in required:
        if col not in df.columns:
            raise ValueError(
                f"Variable absente : {col}"
            )


    # ============================================================
    # 2. CREATION DU GEODATAFRAME DHS
    # ============================================================

    clusters = gpd.GeoDataFrame(
        df.copy(),
        geometry=gpd.points_from_xy(
            df["longitude"],
            df["latitude"],
        ),
        crs="EPSG:4326",
    )


    # ============================================================
    # 3. CHARGEMENT DE LA FRONTIERE
    # ============================================================

    if not BOUNDARY_FILE.exists():

        raise FileNotFoundError(
            "\nLe fichier de frontière est absent :\n"
            f"{BOUNDARY_FILE}\n\n"
            "Télécharge d'abord le fichier GeoJSON "
            "avec la commande indiquée."
        )


    country = gpd.read_file(
        BOUNDARY_FILE
    ).to_crs(
        "EPSG:4326"
    )


    print(
        f"Frontière chargée : {len(country)} géométrie(s)"
    )


    # ============================================================
    # 4. CALCUL DES ERREURS
    # ============================================================

    clusters["error_viirs"] = (
        clusters["pred_viirs"]
        - clusters["wealth_observed"]
    )

    clusters["error_cnn"] = (
        clusters["pred_cnn"]
        - clusters["wealth_observed"]
    )

    clusters["error_cnn_plus_viirs"] = (
        clusters["pred_cnn_plus_viirs"]
        - clusters["wealth_observed"]
    )


    # ============================================================
    # 5. ECHELLES COMMUNES
    # ============================================================

    wealth_columns = [
        "wealth_observed",
        "pred_viirs",
        "pred_cnn",
        "pred_cnn_plus_viirs",
    ]

    wealth_values = np.concatenate(
        [
            clusters[c].to_numpy()
            for c in wealth_columns
        ]
    )

    wealth_vmin = np.percentile(
        wealth_values,
        2
    )

    wealth_vmax = np.percentile(
        wealth_values,
        98
    )


    error_columns = [
        "error_viirs",
        "error_cnn",
        "error_cnn_plus_viirs",
    ]

    error_values = np.concatenate(
        [
            clusters[c].to_numpy()
            for c in error_columns
        ]
    )

    error_limit = np.percentile(
        np.abs(error_values),
        98
    )


    print(
        f"Echelle richesse : "
        f"{wealth_vmin:.3f} à {wealth_vmax:.3f}"
    )

    print(
        f"Echelle erreur : "
        f"{-error_limit:.3f} à {error_limit:.3f}"
    )


    # ============================================================
    # 6. LIMITES DE LA CARTE
    # ============================================================

    minx, miny, maxx, maxy = (
        country.total_bounds
    )

    margin = 0.15


    # ============================================================
    # 7. FONCTION DE BASE
    # ============================================================

    def prepare_axis(ax):

        country.plot(
            ax=ax,
            facecolor="whitesmoke",
            edgecolor="black",
            linewidth=0.8,
        )

        ax.set_xlim(
            minx - margin,
            maxx + margin,
        )

        ax.set_ylim(
            miny - margin,
            maxy + margin,
        )

        ax.set_xlabel(
            "Longitude"
        )

        ax.set_ylabel(
            "Latitude"
        )

        ax.grid(
            alpha=0.15
        )


    # ============================================================
    # 8. PLANCHE COMPARATIVE DE RICHESSE
    # ============================================================

    print(
        "\nCréation de la planche richesse..."
    )

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(13, 14),
        constrained_layout=True,
    )

    axes = axes.ravel()


    wealth_maps = [
        (
            "wealth_observed",
            "(A) Richesse observée",
        ),
        (
            "pred_viirs",
            "(B) Prédiction VIIRS",
        ),
        (
            "pred_cnn",
            "(C) Prédiction CNN Sentinel-2",
        ),
        (
            "pred_cnn_plus_viirs",
            "(D) Prédiction CNN + VIIRS",
        ),
    ]


    scatter_reference = None


    for ax, (
        column,
        title,
    ) in zip(
        axes,
        wealth_maps,
    ):

        prepare_axis(ax)

        scatter_reference = ax.scatter(
            clusters.geometry.x,
            clusters.geometry.y,
            c=clusters[column],
            s=22,
            vmin=wealth_vmin,
            vmax=wealth_vmax,
            alpha=0.85,
            zorder=3,
        )

        ax.set_title(
            title,
            fontsize=12,
        )


    colorbar = fig.colorbar(
        scatter_reference,
        ax=axes.tolist(),
        shrink=0.72,
        pad=0.02,
    )

    colorbar.set_label(
        "Indice de richesse"
    )


    fig.suptitle(
        "Richesse observée et prédite "
        "aux clusters DHS – Côte d’Ivoire, 2021",
        fontsize=15,
    )


    wealth_output = (
        OUTPUT_DIR
        / "figure_wealth_comparison.png"
    )

    fig.savefig(
        wealth_output,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


    print(
        f"Créée : {wealth_output.name}"
    )


    # ============================================================
    # 9. PLANCHE DES ERREURS
    # ============================================================

    print(
        "Création de la planche des erreurs..."
    )

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(18, 7),
        constrained_layout=True,
    )


    error_maps = [
        (
            "error_viirs",
            "(A) VIIRS",
        ),
        (
            "error_cnn",
            "(B) CNN Sentinel-2",
        ),
        (
            "error_cnn_plus_viirs",
            "(C) CNN + VIIRS",
        ),
    ]


    scatter_reference = None


    for ax, (
        column,
        title,
    ) in zip(
        axes,
        error_maps,
    ):

        prepare_axis(ax)

        scatter_reference = ax.scatter(
            clusters.geometry.x,
            clusters.geometry.y,
            c=clusters[column],
            s=22,
            cmap="coolwarm",
            vmin=-error_limit,
            vmax=error_limit,
            alpha=0.85,
            zorder=3,
        )

        ax.set_title(
            title,
            fontsize=12,
        )


    colorbar = fig.colorbar(
        scatter_reference,
        ax=axes.tolist(),
        shrink=0.72,
        pad=0.02,
    )

    colorbar.set_label(
        "Erreur (prédiction − observation)"
    )


    fig.suptitle(
        "Distribution spatiale des erreurs "
        "de prédiction – Côte d’Ivoire, 2021",
        fontsize=15,
    )


    error_output = (
        OUTPUT_DIR
        / "figure_prediction_errors.png"
    )

    fig.savefig(
        error_output,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)


    print(
        f"Créée : {error_output.name}"
    )


    # ============================================================
    # 10. CARTES INDIVIDUELLES
    # ============================================================

    print(
        "Création des cartes individuelles..."
    )


    for column, title in wealth_maps:

        fig, ax = plt.subplots(
            figsize=(8, 9)
        )

        prepare_axis(ax)

        scatter = ax.scatter(
            clusters.geometry.x,
            clusters.geometry.y,
            c=clusters[column],
            s=28,
            vmin=wealth_vmin,
            vmax=wealth_vmax,
            alpha=0.85,
            zorder=3,
        )

        fig.colorbar(
            scatter,
            ax=ax,
            shrink=0.75,
            label="Indice de richesse",
        )

        ax.set_title(title)

        fig.tight_layout()

        filename = (
            column
            + "_publication.png"
        )

        fig.savefig(
            OUTPUT_DIR / filename,
            dpi=300,
            bbox_inches="tight",
        )

        plt.close(fig)


    # ============================================================
    # 11. FIN
    # ============================================================

    print(
        "\n" + "=" * 72
    )

    print(
        "ETAPE 09B TERMINEE AVEC SUCCES"
    )

    print(
        "=" * 72
    )

    print(
        f"\nCartes disponibles dans :\n"
        f"{OUTPUT_DIR}"
    )

    print(
        "\nRappel : ces figures représentent "
        "les clusters DHS et non une surface "
        "continue de richesse."
    )

if __name__ == "__main__":
    main()
