from src.utils.paths import DATA_DIR
from pathlib import Path

import pandas as pd


# Racine du projet
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Base finale DHS + GPS + VIIRS
DATA_FILE = (
    DATA_DIR
    / "processed"
    / "dhs"
    / "DHS_CI_2021_richesse_GPS_VIIRS_merged.csv"
)


def load_data():
    """Charge la base analytique principale."""
    assert DATA_FILE.exists(), f"Fichier introuvable : {DATA_FILE}"
    return pd.read_csv(DATA_FILE)


def test_fichier_existe():
    """Vérifie que la base principale existe."""
    assert DATA_FILE.exists()


def test_nombre_clusters():
    """La base doit contenir les 539 clusters DHS."""
    df = load_data()
    assert len(df) == 539


def test_colonnes_essentielles():
    """Vérifie la présence des variables indispensables."""
    df = load_data()

    colonnes = [
        "cluster",
        "wealth_mean",
        "latitude",
        "longitude",
        "urban_rural",
    ]

    for colonne in colonnes:
        assert colonne in df.columns, f"Colonne manquante : {colonne}"


def test_clusters_uniques():
    """Un cluster ne doit apparaître qu'une seule fois."""
    df = load_data()
    assert df["cluster"].duplicated().sum() == 0


def test_coordonnees_non_manquantes():
    """Tous les clusters doivent avoir des coordonnées GPS."""
    df = load_data()

    assert df["latitude"].notna().all()
    assert df["longitude"].notna().all()


def test_coordonnees_valides():
    """Vérifie que latitude et longitude ont des valeurs possibles."""
    df = load_data()

    assert df["latitude"].between(-90, 90).all()
    assert df["longitude"].between(-180, 180).all()


def test_richesse_non_manquante():
    """Chaque cluster doit avoir une valeur cible de richesse."""
    df = load_data()
    assert df["wealth_mean"].notna().all()


def test_viirs_disponible():
    """Vérifie que le prédicteur VIIRS principal est disponible."""
    df = load_data()

    colonne_viirs = "viirs_2u5r_mean"

    assert colonne_viirs in df.columns
    assert df[colonne_viirs].notna().all()