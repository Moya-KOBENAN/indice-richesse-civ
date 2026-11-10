"""Découpage en tuiles et construction de l'indice EHCVM."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.feature_sets import cnn_file, parse_feature_set
from src.utils.imagery import image_id, split_tiles

ROOT = Path(__file__).resolve().parents[2]


def load_script(relative):
    """Importer un script d'étape (nom commençant par un chiffre) sans l'exécuter."""
    spec = importlib.util.spec_from_file_location("stage", ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_tiles_cover_image_and_skip_empty():
    image = np.full((500, 500, 3), 120, dtype=np.uint8)
    assert len(split_tiles(image, 100)) == 25
    image[:100, :100] = 0
    assert len(split_tiles(image, 100)) == 24
    assert len(split_tiles(image, 0)) == 1


def test_image_ids():
    assert image_id(Path("cluster_012_sentinel2_2021.tif"), "cluster") == 12
    assert image_id(Path("grid_3218_sentinel2_2021.tif"), "grid") == 3218


def test_feature_set_names():
    assert parse_feature_set("cnn_imagenet_tiles+viirs") == (["cnn_imagenet_tiles"], "viirs_2u5r_mean")
    assert parse_feature_set("viirs5") == ([], "viirs_5all_mean")
    assert cnn_file("grid", "cnn_national_tiles").name == "grid_cnn_features_national_tiles.csv"


def test_weighted_pca_orientation_and_quintiles():
    stage = load_script("src/data/ehcvm/02_build_wealth_index.py")
    rng = np.random.default_rng(0)
    latent = rng.normal(size=500)
    X = np.column_stack([(latent + rng.normal(scale=0.5, size=500) > 0).astype(float) for _ in range(6)])
    weights = rng.uniform(0.5, 2, 500)
    score, loading, explained, kept = stage.weighted_pca_first_component(X, weights)
    assert abs(np.corrcoef(score, latent)[0, 1]) > 0.8
    assert explained[0] > explained[1] and kept.all()
    quintiles = stage.weighted_quintiles(score, weights)
    assert set(quintiles) == {1, 2, 3, 4, 5}


def test_rare_housing_categories_are_grouped():
    stage = load_script("src/data/ehcvm/02_build_wealth_index.py")
    frame = pd.DataFrame({"s11q20": [1] * 98 + [2, np.nan]})
    dummies = stage.housing_dummies(frame, ["s11q20"], np.ones(100), 0.05)
    assert list(dummies.columns) == ["s11q20_1", "s11q20_autre"]
