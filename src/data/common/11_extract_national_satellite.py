# Résolution des chemins indépendante du répertoire de lancement.
from pathlib import Path
import sys

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.utils.paths import (
    COMMON_PROCESSED_DIR,
    COMMON_TABLES_DIR,
    NATIONAL_SATELLITE_DIR,
    PROJECT_ROOT,
    ensure_layout,
)

ensure_layout('common')

import io
import time
import zipfile

import ee
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import requests


# ============================================================
# PROJET INDICE DE RICHESSE - COTE D'IVOIRE
# ETAPE 16.3 : EXTRACTION SENTINEL-2 NATIONALE
# GRILLE DE PREDICTION - 3218 POINTS
# ============================================================

from src.utils.config import load_config

PROJECT_ID = load_config()["earth_engine"]["project_id"]

BASE_DIR = PROJECT_ROOT

INPUT_FILE = (
    COMMON_PROCESSED_DIR / 'cote_ivoire_prediction_grid.geojson'
)

OUTPUT_DIR = (
    NATIONAL_SATELLITE_DIR
)

LOG_FILE = (
    COMMON_TABLES_DIR / 'national_satellite_download_log.csv'
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

LOG_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# PARAMETRES
# ============================================================

PATCH_RADIUS_METERS = 2500
SCALE_METERS = 10

START_DATE = "2021-01-01"
END_DATE = "2022-01-01"

CLOUD_THRESHOLD = 40
FALLBACK_CLOUD_THRESHOLD = 80
MINIMUM_IMAGES = 3

PAUSE_SECONDS = 0.5


# ============================================================
# 1. INITIALISATION EARTH ENGINE
# ============================================================

print("=" * 72)
print("ETAPE 16.3 - EXTRACTION SENTINEL-2 NATIONALE")
print("=" * 72)

ee.Initialize(
    project=PROJECT_ID
)

print("Connexion Earth Engine : OK")


# ============================================================
# 2. CHARGEMENT DE LA GRILLE NATIONALE
# ============================================================

if not INPUT_FILE.exists():

    raise FileNotFoundError(
        f"Grille nationale introuvable : {INPUT_FILE}"
    )


grid = gpd.read_file(
    INPUT_FILE
)


required_columns = [
    "grid_id",
    "latitude",
    "longitude",
]


missing_columns = [
    column
    for column in required_columns
    if column not in grid.columns
]


if missing_columns:

    raise ValueError(
        f"Colonnes manquantes : {missing_columns}"
    )


if grid["grid_id"].duplicated().any():

    raise ValueError(
        "Des grid_id dupliqués ont été détectés."
    )


if grid[
    ["latitude", "longitude"]
].isna().any().any():

    raise ValueError(
        "Des coordonnées sont manquantes."
    )


grid = grid.sort_values(
    "grid_id"
).reset_index(
    drop=True
)


print(
    "Points dans la grille :",
    len(grid)
)

print(
    "Points à traiter :",
    len(grid)
)


# ============================================================
# 3. MASQUE SENTINEL-2
# ============================================================

def mask_s2(image):

    scl = image.select(
        "SCL"
    )

    mask = (
        scl.neq(0)
        .And(
            scl.neq(1)
        )
        .And(
            scl.neq(3)
        )
        .And(
            scl.neq(8)
        )
        .And(
            scl.neq(9)
        )
        .And(
            scl.neq(10)
        )
        .And(
            scl.neq(11)
        )
    )

    return (
        image
        .updateMask(mask)
        .select(
            [
                "B4",
                "B3",
                "B2",
            ]
        )
        .divide(10000)
    )


# ============================================================
# 4. CREATION DU COMPOSITE LOCAL
# ============================================================

def create_local_composite(region):

    collection = (
        ee.ImageCollection(
            "COPERNICUS/S2_SR_HARMONIZED"
        )
        .filterDate(
            START_DATE,
            END_DATE
        )
        .filterBounds(
            region
        )
        .filter(
            ee.Filter.lt(
                "CLOUDY_PIXEL_PERCENTAGE",
                CLOUD_THRESHOLD
            )
        )
    )


    n_images = (
        collection
        .size()
        .getInfo()
    )


    # --------------------------------------------------------
    # Fallback si moins de 3 images
    # --------------------------------------------------------

    if n_images < MINIMUM_IMAGES:

        collection = (
            ee.ImageCollection(
                "COPERNICUS/S2_SR_HARMONIZED"
            )
            .filterDate(
                START_DATE,
                END_DATE
            )
            .filterBounds(
                region
            )
            .filter(
                ee.Filter.lt(
                    "CLOUDY_PIXEL_PERCENTAGE",
                    FALLBACK_CLOUD_THRESHOLD
                )
            )
        )

        n_images = (
            collection
            .size()
            .getInfo()
        )


    if n_images == 0:

        return None, 0


    composite = (
        collection
        .map(mask_s2)
        .median()
    )


    return (
        composite,
        n_images
    )


# ============================================================
# 5. CONTROLE QUALITE DES GEOTIFF
# ============================================================

def check_quality(file_path):

    try:

        with rasterio.open(
            file_path
        ) as src:

            data = src.read(
                masked=True
            )

            width = src.width
            height = src.height
            bands = src.count


            if data.count() == 0:

                return {
                    "valid": False,
                    "width": width,
                    "height": height,
                    "bands": bands,
                    "pct_non_zero": 0.0,
                    "mean_reflectance": 0.0,
                }


            values = (
                data.compressed()
            )


            if len(values) == 0:

                return {
                    "valid": False,
                    "width": width,
                    "height": height,
                    "bands": bands,
                    "pct_non_zero": 0.0,
                    "mean_reflectance": 0.0,
                }


            pct_non_zero = (
                100
                * np.count_nonzero(
                    values
                )
                / len(values)
            )


            mean_reflectance = float(
                np.mean(
                    values
                )
            )


            valid = (
                bands == 3
                and width >= 450
                and height >= 450
                and pct_non_zero >= 90
                and mean_reflectance > 0
            )


            return {
                "valid": valid,
                "width": width,
                "height": height,
                "bands": bands,
                "pct_non_zero": pct_non_zero,
                "mean_reflectance": mean_reflectance,
            }


    except Exception:

        return {
            "valid": False,
            "width": 0,
            "height": 0,
            "bands": 0,
            "pct_non_zero": 0.0,
            "mean_reflectance": 0.0,
        }


# ============================================================
# 6. SAUVEGARDE DU JOURNAL
# ============================================================

def save_log(results):

    results_df = pd.DataFrame(
        results
    )

    results_df.to_csv(
        LOG_FILE,
        index=False
    )


# ============================================================
# 7. TRAITEMENT D'UN POINT DE LA GRILLE
# ============================================================

def process_grid_point(row):

    grid_id = int(
        row["grid_id"]
    )

    lat = float(
        row["latitude"]
    )

    lon = float(
        row["longitude"]
    )


    output_file = (
        OUTPUT_DIR
        / (
            f"grid_{grid_id:04d}"
            "_sentinel2_2021.tif"
        )
    )


    print()
    print(
        "------------------------------------"
    )

    print(
        f"Grid ID : {grid_id}"
    )

    print(
        f"Coordonnées : "
        f"{lat:.4f}, {lon:.4f}"
    )


    # ========================================================
    # REPRISE AUTOMATIQUE
    # ========================================================

    if output_file.exists():

        quality = check_quality(
            output_file
        )


        if quality["valid"]:

            print(
                "Fichier déjà présent et valide "
                "-> IGNORE"
            )

            return {
                "grid_id": grid_id,
                "status": "DEJA_OK",
                "n_images": np.nan,
                "width": quality["width"],
                "height": quality["height"],
                "bands": quality["bands"],
                "pct_non_zero":
                    quality["pct_non_zero"],
                "mean_reflectance":
                    quality["mean_reflectance"],
                "error": "",
            }


        print(
            "Fichier existant invalide "
            "-> nouveau téléchargement"
        )

        output_file.unlink(
            missing_ok=True
        )


    # ========================================================
    # REGION 5 KM x 5 KM
    # ========================================================

    point = ee.Geometry.Point(
        [
            lon,
            lat,
        ]
    )


    region = (
        point
        .buffer(
            PATCH_RADIUS_METERS
        )
        .bounds()
    )


    composite, n_images = (
        create_local_composite(
            region
        )
    )


    print(
        "Images Sentinel-2 :",
        n_images
    )


    if composite is None:

        return {
            "grid_id": grid_id,
            "status": "ECHEC_AUCUNE_IMAGE",
            "n_images": 0,
            "width": 0,
            "height": 0,
            "bands": 0,
            "pct_non_zero": 0.0,
            "mean_reflectance": 0.0,
            "error":
                "Aucune image Sentinel-2",
        }


    # ========================================================
    # URL EARTH ENGINE
    # ========================================================

    url = composite.getDownloadURL(
        {
            "region": region,
            "scale": SCALE_METERS,
            "crs": "EPSG:4326",
            "format": "GEO_TIFF",
        }
    )


    response = requests.get(
        url,
        timeout=180
    )


    response.raise_for_status()


    # ========================================================
    # ECRITURE DU GEOTIFF
    # ========================================================

    if response.content[:2] == b"PK":

        with zipfile.ZipFile(
            io.BytesIO(
                response.content
            )
        ) as archive:

            tif_files = [
                name
                for name
                in archive.namelist()
                if name
                .lower()
                .endswith(".tif")
            ]


            if not tif_files:

                raise RuntimeError(
                    "Aucun GeoTIFF reçu."
                )


            with archive.open(
                tif_files[0]
            ) as source:

                output_file.write_bytes(
                    source.read()
                )


    else:

        output_file.write_bytes(
            response.content
        )


    # ========================================================
    # CONTROLE QUALITE
    # ========================================================

    quality = check_quality(
        output_file
    )


    print(
        f"Dimensions : "
        f"{quality['width']} x "
        f"{quality['height']}"
    )


    print(
        f"Pixels non nuls : "
        f"{quality['pct_non_zero']:.2f} %"
    )


    print(
        f"Réflectance moyenne : "
        f"{quality['mean_reflectance']:.4f}"
    )


    if quality["valid"]:

        print(
            "QUALITE : OK"
        )

        status = "OK"


    else:

        print(
            "QUALITE : ECHEC"
        )

        status = (
            "ECHEC_QUALITE"
        )


    return {
        "grid_id": grid_id,
        "status": status,
        "n_images": n_images,
        "width": quality["width"],
        "height": quality["height"],
        "bands": quality["bands"],
        "pct_non_zero":
            quality["pct_non_zero"],
        "mean_reflectance":
            quality["mean_reflectance"],
        "error": "",
    }


# ============================================================
# 8. EXTRACTION NATIONALE
# ============================================================

print()
print("=" * 72)
print(
    "EXTRACTION SENTINEL-2 "
    "- GRILLE NATIONALE"
)
print(
    "PATCH : ENVIRON 5 KM x 5 KM"
)
print(
    "TOTAL :",
    len(grid),
    "POINTS"
)
print("=" * 72)


results = []

total = len(grid)


for position, (_, row) in enumerate(
    grid.iterrows(),
    start=1
):

    grid_id = int(
        row["grid_id"]
    )


    print()
    print(
        f"[{position}/{total}] "
        f"Traitement du point "
        f"{grid_id}"
    )


    try:

        result = (
            process_grid_point(
                row
            )
        )


    except Exception as error:

        print(
            "ERREUR :",
            str(error)
        )


        result = {
            "grid_id": grid_id,
            "status": "ERREUR",
            "n_images": np.nan,
            "width": 0,
            "height": 0,
            "bands": 0,
            "pct_non_zero": 0.0,
            "mean_reflectance": 0.0,
            "error": str(error),
        }


    results.append(
        result
    )


    # --------------------------------------------------------
    # Sauvegarde après chaque point
    # --------------------------------------------------------

    save_log(
        results
    )


    # --------------------------------------------------------
    # Petite pause entre les requêtes
    # --------------------------------------------------------

    time.sleep(
        PAUSE_SECONDS
    )


# ============================================================
# 9. BILAN FINAL
# ============================================================

results_df = pd.DataFrame(
    results
)


successful = results_df[
    results_df[
        "status"
    ].isin(
        [
            "OK",
            "DEJA_OK",
        ]
    )
]


problems = results_df[
    ~results_df[
        "status"
    ].isin(
        [
            "OK",
            "DEJA_OK",
        ]
    )
]


print()
print()
print("=" * 72)
print("BILAN FINAL")
print("=" * 72)


print(
    "Nombre total :",
    len(results_df)
)


print(
    "Images valides :",
    len(successful)
)


print(
    "Problèmes :",
    len(problems)
)


if len(problems) > 0:

    print()
    print(
        "Points à vérifier :"
    )

    print(
        problems[
            [
                "grid_id",
                "status",
                "error",
            ]
        ].to_string(
            index=False
        )
    )


print()
print(
    "Journal sauvegardé :"
)

print(
    LOG_FILE
)


print()
print(
    "Images sauvegardées :"
)

print(
    OUTPUT_DIR
)


print()
print("=" * 72)


if len(successful) == total:

    print(
        "EXTRACTION COMPLETE : "
        f"{len(successful)}/"
        f"{total} POINTS VALIDES"
    )


else:

    print(
        "EXTRACTION TERMINEE : "
        f"{len(successful)}/"
        f"{total} POINTS VALIDES "
        f"- {len(problems)} "
        "PROBLEME(S)"
    )


print("=" * 72)