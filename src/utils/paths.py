"""Chemins du projet : unique source de vérité pour l'arborescence.

Aucun script ne construit de chemin à partir du répertoire courant ; tout part
de PROJECT_ROOT, ce qui rend les étapes exécutables depuis n'importe où.
"""
from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONFIG_DIR = PROJECT_ROOT / "config"

DATA_DIR = Path(os.environ.get("RICHESSE_DATA_DIR", str(PROJECT_ROOT / "data"))).expanduser().resolve()
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

COMMON_PROCESSED_DIR = PROCESSED_DIR / "common"
DHS_PROCESSED_DIR = PROCESSED_DIR / "dhs"
DHS_INTERIM_DIR = INTERIM_DIR / "dhs"
EHCVM_INTERIM_DIR = INTERIM_DIR / "ehcvm"
EHCVM_PROCESSED_DIR = PROCESSED_DIR / "ehcvm"

DHS_SATELLITE_DIR = RAW_DIR / "satellite"
NATIONAL_SATELLITE_DIR = RAW_DIR / "satellite_national"

OUTPUTS_DIR = Path(os.environ.get("RICHESSE_OUTPUTS_DIR", str(PROJECT_ROOT / "outputs"))).expanduser().resolve()

COMMON_OUTPUTS_DIR = OUTPUTS_DIR / "common"
COMMON_MODELS_DIR = COMMON_OUTPUTS_DIR / "models"
COMMON_TABLES_DIR = COMMON_OUTPUTS_DIR / "tables"
COMMON_FIGURES_DIR = COMMON_OUTPUTS_DIR / "figures"

DHS_OUTPUTS_DIR = OUTPUTS_DIR / "dhs"
DHS_MODELS_DIR = DHS_OUTPUTS_DIR / "models"
DHS_TABLES_DIR = DHS_OUTPUTS_DIR / "tables"
DHS_FIGURES_DIR = DHS_OUTPUTS_DIR / "figures"
DHS_MAPS_DIR = DHS_OUTPUTS_DIR / "maps"

EHCVM_OUTPUTS_DIR = OUTPUTS_DIR / "ehcvm"
EHCVM_MODELS_DIR = EHCVM_OUTPUTS_DIR / "models"
EHCVM_TABLES_DIR = EHCVM_OUTPUTS_DIR / "tables"
EHCVM_FIGURES_DIR = EHCVM_OUTPUTS_DIR / "figures"
EHCVM_MAPS_DIR = EHCVM_OUTPUTS_DIR / "maps"

COMPARISON_OUTPUTS_DIR = OUTPUTS_DIR / "comparison"

REPORTS_DIR = PROJECT_ROOT / "reports"
DOCS_DIR = PROJECT_ROOT / "docs"
BOUNDARY_FILE = RAW_DIR / "cote_ivoire_boundary.geojson"

EHCVM_REPORTS_DIR = REPORTS_DIR / "ehcvm"
EHCVM_RAW_DIR = RAW_DIR / "ehcvm"
EHCVM_SATELLITE_DIR = EHCVM_RAW_DIR / "satellite"


def ensure_layout(scope):
    """Créer les répertoires d'une branche seulement lors de son exécution."""
    directories = {
        "dhs": [DHS_INTERIM_DIR, DHS_PROCESSED_DIR, DHS_MODELS_DIR, DHS_TABLES_DIR,
                DHS_FIGURES_DIR, DHS_MAPS_DIR],
        "common": [COMMON_PROCESSED_DIR, COMMON_MODELS_DIR,
                   COMMON_TABLES_DIR, COMMON_FIGURES_DIR,
                   COMMON_OUTPUTS_DIR / "maps"],
        "ehcvm": [EHCVM_INTERIM_DIR, EHCVM_PROCESSED_DIR, EHCVM_SATELLITE_DIR, EHCVM_MODELS_DIR, EHCVM_TABLES_DIR,
                  EHCVM_FIGURES_DIR, EHCVM_MAPS_DIR, EHCVM_REPORTS_DIR],
    }
    for directory in directories[scope]:
        directory.mkdir(parents=True, exist_ok=True)


SURVEYS = ("dhs", "ehcvm")


def survey_dirs(survey):
    """Répertoires d'une enquête, pour les étapes communes à DHS et EHCVM."""
    if survey not in SURVEYS:
        raise ValueError(f"Enquête inconnue : {survey}. Choix : {SURVEYS}")
    outputs = OUTPUTS_DIR / survey
    return {
        "interim": INTERIM_DIR / survey,
        "processed": PROCESSED_DIR / survey,
        "satellite": DHS_SATELLITE_DIR if survey == "dhs" else EHCVM_SATELLITE_DIR,
        "tables": outputs / "tables",
        "models": outputs / "models",
        "figures": outputs / "figures",
        "maps": outputs / "maps",
    }


def survey_clusters_file(survey):
    """Table analytique d'une enquête : une ligne par grappe, cible + GPS + VIIRS."""
    if survey == "dhs":
        return DHS_PROCESSED_DIR / "DHS_CI_2021_richesse_GPS_VIIRS_merged.csv"
    return EHCVM_PROCESSED_DIR / "ehcvm_clusters_wealth_gps_viirs.csv"
