from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CNN_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "dhs"
    / "cnn_features_resnet50.csv"
)


def load_features():
    assert CNN_FILE.exists(), f"Fichier introuvable : {CNN_FILE}"
    return pd.read_csv(CNN_FILE)


def test_fichier_features_existe():
    assert CNN_FILE.exists()


def test_nombre_clusters_cnn():
    df = load_features()
    assert len(df) == 539


def test_clusters_cnn_uniques():
    df = load_features()
    assert df["cluster"].duplicated().sum() == 0


def test_nombre_features_cnn():
    df = load_features()

    feature_cols = [
        c for c in df.columns
        if c.startswith("cnn_")
    ]

    assert len(feature_cols) == 2048


def test_features_sans_nan():
    df = load_features()

    feature_cols = [
        c for c in df.columns
        if c.startswith("cnn_")
    ]

    assert df[feature_cols].notna().all().all()


def test_features_finies():
    df = load_features()

    feature_cols = [
        c for c in df.columns
        if c.startswith("cnn_")
    ]

    values = df[feature_cols].to_numpy()

    assert np.isfinite(values).all()


def test_features_variables():
    df = load_features()

    feature_cols = [
        c for c in df.columns
        if c.startswith("cnn_")
    ]

    variances = df[feature_cols].var()

    assert (variances > 0).all()


def test_clusters_correspondent_base_principale():

    features = load_features()

    main_file = (
        PROJECT_ROOT
        / "data"
        / "processed"
    / "dhs"
        / "DHS_CI_2021_richesse_GPS_VIIRS_merged.csv"
    )

    main = pd.read_csv(main_file)

    assert set(features["cluster"]) == set(main["cluster"])