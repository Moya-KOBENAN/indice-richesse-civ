# Résolution des chemins indépendante du répertoire de lancement.
from pathlib import Path
import sys

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.utils.paths import (
    DHS_FIGURES_DIR,
    DHS_MAPS_DIR,
    DHS_TABLES_DIR,
    PROJECT_ROOT,
    ensure_layout,
)

ensure_layout('dhs')


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 09 - CARTOGRAPHIE DES RESULTATS
#
# Cartes diagnostiques au niveau des clusters DHS :
# 1. richesse observée
# 2. richesse prédite par VIIRS
# 3. richesse prédite par CNN
# 4. richesse prédite par CNN + VIIRS
# 5. erreurs de prédiction
#
# Les prédictions utilisées proviennent de la validation
# spatiale par blocs réguliers de 100 km (étape 08B).
# ============================================================


# ============================================================
# 1. PARAMETRES
# ============================================================


PREDICTIONS_FILE = (
    DHS_TABLES_DIR / 'spatial_block_robustness_predictions.csv'
)

FIGURES_DIR = (
    DHS_FIGURES_DIR
)

MAPS_DIR = (
    DHS_MAPS_DIR
)

TABLES_DIR = (
    DHS_TABLES_DIR
)

FIGURES_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MAPS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

TABLES_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# 2. CHARGEMENT
# ============================================================

print("=" * 72)
print("ETAPE 09 - CARTOGRAPHIE DES RESULTATS")
print("=" * 72)

df = pd.read_csv(
    PREDICTIONS_FILE
)

print(
    f"Base de prédictions : {df.shape}"
)


# ============================================================
# 3. CONTROLES
# ============================================================

required_columns = [
    "cluster",
    "latitude",
    "longitude",
    "wealth_observed",
    "pred_viirs",
    "pred_cnn",
    "pred_cnn_plus_viirs",
]

for column in required_columns:

    if column not in df.columns:

        raise ValueError(
            f"Variable absente : {column}"
        )


if len(df) != 539:

    raise ValueError(
        f"539 clusters attendus, "
        f"{len(df)} obtenus."
    )


if df["cluster"].duplicated().any():

    raise ValueError(
        "Doublons de clusters détectés."
    )


if df[
    required_columns
].isna().any().any():

    raise ValueError(
        "Valeurs manquantes détectées."
    )


numeric_columns = [
    "latitude",
    "longitude",
    "wealth_observed",
    "pred_viirs",
    "pred_cnn",
    "pred_cnn_plus_viirs",
]


if not np.isfinite(
    df[numeric_columns].to_numpy()
).all():

    raise ValueError(
        "Valeurs infinies détectées."
    )


print(
    "Contrôles des données : OK"
)


# ============================================================
# 4. CALCUL DES ERREURS
# ============================================================

df["error_viirs"] = (
    df["pred_viirs"]
    - df["wealth_observed"]
)


df["error_cnn"] = (
    df["pred_cnn"]
    - df["wealth_observed"]
)


df["error_cnn_plus_viirs"] = (
    df["pred_cnn_plus_viirs"]
    - df["wealth_observed"]
)


df["abs_error_viirs"] = (
    df["error_viirs"].abs()
)


df["abs_error_cnn"] = (
    df["error_cnn"].abs()
)


df["abs_error_cnn_plus_viirs"] = (
    df["error_cnn_plus_viirs"].abs()
)


# ============================================================
# 5. LIMITES COMMUNES POUR LES CARTES DE RICHESSE
# ============================================================

wealth_values = np.concatenate(
    [
        df["wealth_observed"].to_numpy(),
        df["pred_viirs"].to_numpy(),
        df["pred_cnn"].to_numpy(),
        df["pred_cnn_plus_viirs"].to_numpy(),
    ]
)


wealth_vmin = float(
    np.nanpercentile(
        wealth_values,
        2
    )
)


wealth_vmax = float(
    np.nanpercentile(
        wealth_values,
        98
    )
)


# Même échelle pour toutes les cartes d'erreur.

all_errors = np.concatenate(
    [
        df["error_viirs"].to_numpy(),
        df["error_cnn"].to_numpy(),
        df["error_cnn_plus_viirs"].to_numpy(),
    ]
)


error_limit = float(
    np.nanpercentile(
        np.abs(all_errors),
        98
    )
)


print(
    f"Echelle richesse : "
    f"{wealth_vmin:.3f} à "
    f"{wealth_vmax:.3f}"
)

print(
    f"Echelle erreur : "
    f"-{error_limit:.3f} à "
    f"+{error_limit:.3f}"
)


# ============================================================
# 6. FONCTION CARTE DE RICHESSE
# ============================================================

def plot_wealth_map(
    data,
    column,
    title,
    filename,
):

    fig, ax = plt.subplots(
        figsize=(8, 9)
    )


    scatter = ax.scatter(
        data["longitude"],
        data["latitude"],
        c=data[column],
        s=28,
        vmin=wealth_vmin,
        vmax=wealth_vmax,
        alpha=0.85,
    )


    colorbar = fig.colorbar(
        scatter,
        ax=ax,
        shrink=0.75,
    )


    colorbar.set_label(
        "Indice de richesse"
    )


    ax.set_title(
        title
    )


    ax.set_xlabel(
        "Longitude"
    )


    ax.set_ylabel(
        "Latitude"
    )


    ax.set_aspect(
        "equal",
        adjustable="box"
    )


    ax.grid(
        alpha=0.20
    )


    fig.tight_layout()


    output_file = (
        MAPS_DIR
        / filename
    )


    fig.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
    )


    plt.close(fig)


    print(
        f"Carte créée : {output_file.name}"
    )


# ============================================================
# 7. FONCTION CARTE DES ERREURS
# ============================================================

def plot_error_map(
    data,
    column,
    title,
    filename,
):

    fig, ax = plt.subplots(
        figsize=(8, 9)
    )


    scatter = ax.scatter(
        data["longitude"],
        data["latitude"],
        c=data[column],
        s=28,
        vmin=-error_limit,
        vmax=error_limit,
        cmap="coolwarm",
        alpha=0.85,
    )


    colorbar = fig.colorbar(
        scatter,
        ax=ax,
        shrink=0.75,
    )


    colorbar.set_label(
        "Erreur : prédiction - observation"
    )


    ax.set_title(
        title
    )


    ax.set_xlabel(
        "Longitude"
    )


    ax.set_ylabel(
        "Latitude"
    )


    ax.set_aspect(
        "equal",
        adjustable="box"
    )


    ax.grid(
        alpha=0.20
    )


    fig.tight_layout()


    output_file = (
        MAPS_DIR
        / filename
    )


    fig.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
    )


    plt.close(fig)


    print(
        f"Carte créée : {output_file.name}"
    )


# ============================================================
# 8. CARTES DE RICHESSE
# ============================================================

print(
    "\nCréation des cartes de richesse..."
)


plot_wealth_map(
    df,
    "wealth_observed",
    (
        "Indice de richesse observé "
        "- clusters DHS 2021"
    ),
    "01_wealth_observed.png",
)


plot_wealth_map(
    df,
    "pred_viirs",
    (
        "Indice de richesse prédit "
        "- VIIRS"
    ),
    "02_wealth_predicted_viirs.png",
)


plot_wealth_map(
    df,
    "pred_cnn",
    (
        "Indice de richesse prédit "
        "- CNN Sentinel-2"
    ),
    "03_wealth_predicted_cnn.png",
)


plot_wealth_map(
    df,
    "pred_cnn_plus_viirs",
    (
        "Indice de richesse prédit "
        "- CNN Sentinel-2 + VIIRS"
    ),
    "04_wealth_predicted_cnn_viirs.png",
)


# ============================================================
# 9. CARTES DES ERREURS
# ============================================================

print(
    "\nCréation des cartes d'erreur..."
)


plot_error_map(
    df,
    "error_viirs",
    (
        "Erreur de prédiction "
        "- VIIRS"
    ),
    "05_error_viirs.png",
)


plot_error_map(
    df,
    "error_cnn",
    (
        "Erreur de prédiction "
        "- CNN Sentinel-2"
    ),
    "06_error_cnn.png",
)


plot_error_map(
    df,
    "error_cnn_plus_viirs",
    (
        "Erreur de prédiction "
        "- CNN Sentinel-2 + VIIRS"
    ),
    "07_error_cnn_viirs.png",
)


# ============================================================
# 10. OBSERVE VS PREDIT
# ============================================================

models = {
    "VIIRS": "pred_viirs",
    "CNN": "pred_cnn",
    "CNN + VIIRS": "pred_cnn_plus_viirs",
}


for model_name, prediction_column in models.items():

    fig, ax = plt.subplots(
        figsize=(7, 7)
    )


    ax.scatter(
        df["wealth_observed"],
        df[prediction_column],
        s=24,
        alpha=0.65,
    )


    lower = min(
        df["wealth_observed"].min(),
        df[prediction_column].min(),
    )


    upper = max(
        df["wealth_observed"].max(),
        df[prediction_column].max(),
    )


    ax.plot(
        [lower, upper],
        [lower, upper],
        linestyle="--",
    )


    ax.set_xlabel(
        "Richesse observée"
    )


    ax.set_ylabel(
        "Richesse prédite"
    )


    ax.set_title(
        f"Observé vs prédit - {model_name}"
    )


    ax.grid(
        alpha=0.20
    )


    fig.tight_layout()


    safe_name = (
        model_name
        .lower()
        .replace(" ", "_")
        .replace("+", "plus")
    )


    output_file = (
        FIGURES_DIR
        / f"observed_vs_predicted_{safe_name}.png"
    )


    fig.savefig(
        output_file,
        dpi=300,
        bbox_inches="tight",
    )


    plt.close(fig)


    print(
        f"Figure créée : {output_file.name}"
    )


# ============================================================
# 11. TABLE DIAGNOSTIQUE
# ============================================================

diagnostic_columns = [
    "cluster",
    "latitude",
    "longitude",
    "wealth_observed",
    "pred_viirs",
    "pred_cnn",
    "pred_cnn_plus_viirs",
    "error_viirs",
    "error_cnn",
    "error_cnn_plus_viirs",
    "abs_error_viirs",
    "abs_error_cnn",
    "abs_error_cnn_plus_viirs",
]


diagnostic_table = (
    df[
        diagnostic_columns
    ]
    .copy()
)


diagnostic_file = (
    TABLES_DIR
    / "mapping_cluster_diagnostics.csv"
)


diagnostic_table.to_csv(
    diagnostic_file,
    index=False
)


# ============================================================
# 12. CLUSTERS AVEC LES PLUS GRANDES ERREURS
# ============================================================

print(
    "\nPlus grandes erreurs absolues :"
)


for model_name, error_column in {
    "VIIRS": "abs_error_viirs",
    "CNN": "abs_error_cnn",
    "CNN+VIIRS": "abs_error_cnn_plus_viirs",
}.items():

    top_errors = (
        df[
            [
                "cluster",
                "wealth_observed",
                error_column,
            ]
        ]
        .nlargest(
            5,
            error_column
        )
    )


    print(
        f"\n{model_name}"
    )

    print(
        top_errors.to_string(
            index=False
        )
    )


# ============================================================
# 13. FIN
# ============================================================

print(
    "\n" + "=" * 72
)

print(
    "ETAPE 09 TERMINEE AVEC SUCCES"
)

print(
    "=" * 72
)


print(
    "\nLes cartes produites sont des cartes "
    "diagnostiques des clusters DHS."
)

print(
    "Elles ne constituent pas encore une "
    "carte continue de richesse de la "
    "Côte d'Ivoire."
)

print(
    "\nDossier cartes :"
)

print(
    MAPS_DIR
)

print(
    "\nDossier figures :"
)

print(
    FIGURES_DIR
)