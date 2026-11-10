# Résolution des chemins indépendante du répertoire de lancement.
from pathlib import Path
import sys

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.utils.paths import (
    COMMON_PROCESSED_DIR,
    PROJECT_ROOT,
    ensure_layout,
)

ensure_layout('common')


import ee
import geopandas as gpd
import numpy as np
import pandas as pd


# ============================================================
# 10B - PREPARATION DE LA CIBLE VIIRS NATIONALE
# Projet : Indice de richesse - Côte d'Ivoire
#
# Objectif :
# associer à chacun des points de la grille nationale
# une mesure uniforme de luminosité nocturne VIIRS.
# ============================================================


# ============================================================
# 1. PARAMETRES
# ============================================================


GRID_FILE = (
    COMMON_PROCESSED_DIR / 'cote_ivoire_prediction_grid.geojson'
)

OUTPUT_FILE = (
    COMMON_PROCESSED_DIR / 'cote_ivoire_national_grid_viirs.csv'
)

from src.utils.config import load_config

PROJECT_ID = load_config()["earth_engine"]["project_id"]

EXPECTED_POINTS = 3218

YEAR = 2021

RADIUS_METERS = 2500

SCALE_METERS = 500


# ============================================================
# 2. INITIALISATION EARTH ENGINE
# ============================================================

print("=" * 72)
print("10B - PREPARATION VIIRS POUR LA GRILLE NATIONALE")
print("=" * 72)


ee.Initialize(
    project=PROJECT_ID
)

print("Earth Engine : OK")


# ============================================================
# 3. CHARGEMENT DE LA GRILLE
# ============================================================

grid = gpd.read_file(
    GRID_FILE
)


print(
    f"Nombre de points : {len(grid)}"
)


if len(grid) != EXPECTED_POINTS:

    raise ValueError(
        f"{EXPECTED_POINTS} points attendus, "
        f"{len(grid)} trouvés."
    )


if grid.crs is None:

    raise ValueError(
        "CRS absent dans la grille."
    )


grid = grid.to_crs(
    "EPSG:4326"
)


# ============================================================
# 4. IDENTIFIANT UNIQUE
# ============================================================

if "grid_id" not in grid.columns:

    grid["grid_id"] = np.arange(
        1,
        len(grid) + 1
    )


if grid["grid_id"].duplicated().any():

    raise ValueError(
        "Des grid_id sont dupliqués."
    )


# ============================================================
# 5. COORDONNEES
# ============================================================

grid["longitude"] = (
    grid.geometry.x
)

grid["latitude"] = (
    grid.geometry.y
)


# ============================================================
# 6. IMAGE VIIRS 2021
# ============================================================

viirs_collection = (
    ee.ImageCollection(
        "NOAA/VIIRS/DNB/ANNUAL_V21"
    )
    .filterDate(
        f"{YEAR}-01-01",
        f"{YEAR + 1}-01-01"
    )
)


collection_size = (
    viirs_collection
    .size()
    .getInfo()
)


print(
    f"Images VIIRS annuelles trouvées : "
    f"{collection_size}"
)


if collection_size == 0:

    raise ValueError(
        "Aucune image VIIRS trouvée."
    )


viirs_image = (
    viirs_collection
    .select(
        "average_masked"
    )
    .mean()
)


# ============================================================
# 7. CONVERSION DE LA GRILLE EN FEATURECOLLECTION
# ============================================================

features = []


for _, row in grid.iterrows():

    point = ee.Geometry.Point(
        [
            float(row["longitude"]),
            float(row["latitude"])
        ]
    )

    feature = ee.Feature(
        point,
        {
            "grid_id":
                int(row["grid_id"]),

            "longitude":
                float(row["longitude"]),

            "latitude":
                float(row["latitude"]),
        }
    )

    features.append(
        feature
    )


ee_grid = ee.FeatureCollection(
    features
)


print(
    "FeatureCollection Earth Engine créée : OK"
)


# ============================================================
# 8. CALCUL DE LA MOYENNE VIIRS
# ============================================================

def calculate_viirs(feature):

    geometry = (
        feature
        .geometry()
        .buffer(
            RADIUS_METERS
        )
    )

    value = (
        viirs_image
        .reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=geometry,
            scale=SCALE_METERS,
            bestEffort=True,
            maxPixels=1_000_000
        )
        .get(
            "average_masked"
        )
    )

    return feature.set(
        "viirs_mean_2p5km",
        value
    )


ee_result = ee_grid.map(
    calculate_viirs
)


print(
    "Calcul VIIRS configuré : OK"
)


# ============================================================
# 9. RECUPERATION PAR LOTS
# ============================================================

BATCH_SIZE = 400

rows = []


for start in range(
    0,
    EXPECTED_POINTS,
    BATCH_SIZE
):

    end = min(
        start + BATCH_SIZE,
        EXPECTED_POINTS
    )

    batch = (
        ee_result
        .toList(
            BATCH_SIZE,
            start
        )
        .getInfo()
    )


    for item in batch:

        properties = item[
            "properties"
        ]

        rows.append(
            {
                "grid_id":
                    properties.get(
                        "grid_id"
                    ),

                "longitude":
                    properties.get(
                        "longitude"
                    ),

                "latitude":
                    properties.get(
                        "latitude"
                    ),

                "viirs_mean_2p5km":
                    properties.get(
                        "viirs_mean_2p5km"
                    ),
            }
        )


    print(
        f"Points traités : "
        f"{end}/{EXPECTED_POINTS}"
    )


# ============================================================
# 10. CREATION DU TABLEAU
# ============================================================

result = pd.DataFrame(
    rows
)


result = (
    result
    .sort_values("grid_id")
    .reset_index(drop=True)
)


# ============================================================
# 11. CONTROLES
# ============================================================

print()
print("=" * 72)
print("CONTROLE VIIRS NATIONAL")
print("=" * 72)


print(
    f"Nombre de lignes : {len(result)}"
)


duplicates = int(
    result["grid_id"]
    .duplicated()
    .sum()
)


missing = int(
    result["viirs_mean_2p5km"]
    .isna()
    .sum()
)


print(
    f"Doublons grid_id : {duplicates}"
)

print(
    f"VIIRS manquants : {missing}"
)


if len(result) != EXPECTED_POINTS:

    raise ValueError(
        "Nombre incorrect de points."
    )


if duplicates != 0:

    raise ValueError(
        "Des grid_id sont dupliqués."
    )


# ============================================================
# 12. STATISTIQUES
# ============================================================

print()
print(
    "Statistiques VIIRS :"
)

print(
    result[
        "viirs_mean_2p5km"
    ]
    .describe()
)


if missing == 0:

    print(
        "\nValeurs <= 0 :",
        int(
            (
                result[
                    "viirs_mean_2p5km"
                ]
                <= 0
            ).sum()
        )
    )


# ============================================================
# 13. CIBLE LOG1P
# ============================================================

if (
    result["viirs_mean_2p5km"]
    .dropna()
    .lt(0)
    .any()
):

    print(
        "\nATTENTION : "
        "des valeurs VIIRS négatives existent."
    )

    print(
        "La transformation log1p ne sera "
        "pas encore appliquée."
    )

else:

    result[
        "viirs_log1p"
    ] = np.log1p(
        result[
            "viirs_mean_2p5km"
        ]
    )

    print(
        "\nTransformation log1p : OK"
    )


# ============================================================
# 14. SAUVEGARDE
# ============================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)


result.to_csv(
    OUTPUT_FILE,
    index=False
)


print()
print(
    f"Fichier sauvegardé : {OUTPUT_FILE}"
)

print()
print("=" * 72)
print("10B TERMINE AVEC SUCCES")
print("=" * 72)