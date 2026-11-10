# Résolution des chemins indépendante du répertoire de lancement.
from pathlib import Path
import sys

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.utils.paths import (
    DHS_INTERIM_DIR,
    DHS_PROCESSED_DIR,
    PROJECT_ROOT,
    RAW_DIR,
    ensure_layout,
)

ensure_layout('dhs')

import pandas as pd


# ---------------------------------------------------------
# 1. CHEMINS DU PROJET
# ---------------------------------------------------------

ROOT = PROJECT_ROOT

DATA_RAW = RAW_DIR
DATA_PROCESSED = DHS_PROCESSED_DIR

DATA_PROCESSED.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# 2. FICHIERS D'ENTREE
# ---------------------------------------------------------

# Table reconstruite par les étapes 01-02 à partir des fichiers DHS bruts.
wealth_file = DHS_INTERIM_DIR / "DHS_CI_2021_clusters_richesse_GPS.csv"
# Export du script GEE src/data/dhs/00_extract_viirs_dhs_gee.js
viirs_file = DATA_RAW / "DHS_CI_2021_VIIRS_clusters.csv"


# ---------------------------------------------------------
# 3. LECTURE DES DONNEES
# ---------------------------------------------------------

wealth = pd.read_csv(wealth_file)
viirs = pd.read_csv(viirs_file)


# ---------------------------------------------------------
# 4. VERIFICATION RAPIDE
# ---------------------------------------------------------

print("Base richesse :", wealth.shape)
print("Base VIIRS :", viirs.shape)

print("\nColonnes richesse :")
print(wealth.columns.tolist())

print("\nColonnes VIIRS :")
print(viirs.columns.tolist())


# ---------------------------------------------------------
# 5. PREPARATION DE LA CLE DE FUSION
# ---------------------------------------------------------

wealth["cluster"] = wealth["cluster"].astype(int)
viirs["DHSCLUST"] = viirs["DHSCLUST"].astype(int)


# ---------------------------------------------------------
# 6. FUSION
# ---------------------------------------------------------

merged = wealth.merge(
    viirs,
    left_on="cluster",
    right_on="DHSCLUST",
    how="inner",
    validate="one_to_one"
)


# ---------------------------------------------------------
# 7. CONTROLES
# ---------------------------------------------------------

print("\nNombre de clusters après fusion :", len(merged))
if len(merged) != len(wealth):
    raise ValueError(
        f"{len(wealth) - len(merged)} grappes sans VIIRS : fusion incomplète."
    )
print("Nombre de doublons :", merged["cluster"].duplicated().sum())

viirs_cols = [
    "viirs_2u5r_mean",
    "viirs_5all_mean",
    "viirs_2u10r_mean"
]

print(
    "Valeurs VIIRS manquantes :",
    merged[viirs_cols].isna().sum().sum()
)


# ---------------------------------------------------------
# 8. SAUVEGARDE
# ---------------------------------------------------------

output_file = (
    DATA_PROCESSED
    / "DHS_CI_2021_richesse_GPS_VIIRS_merged.csv"
)

merged.to_csv(output_file, index=False)

print("\nFichier sauvegardé ici :")
print(output_file)