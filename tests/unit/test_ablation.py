"""Intégration du tableau d'ablation sur un petit jeu synthétique."""
import numpy as np
import pandas as pd

from src.evaluation.common.compare_feature_sets import BASELINES, evaluate

CONFIG = {"project": {"random_state": 0},
          "model": {"alphas": {"log10_min": -2, "log10_max": 3, "n": 6}},
          "validation": {"n_splits": 4, "n_spatial_folds": 4, "block_size_km": 150,
                         "kmeans_groups": 8, "inner_folds": 3, "bootstrap_iterations": 50}}


def test_evaluate_returns_baselines_and_models(monkeypatch):
    monkeypatch.setattr("src.evaluation.common.compare_feature_sets.alpha_grid",
                        lambda: np.logspace(-2, 3, 6))
    rng = np.random.default_rng(0)
    n = 160
    base = pd.DataFrame({"cluster": range(n), "latitude": rng.uniform(4.5, 10.5, n),
                         "longitude": rng.uniform(-8.5, -2.5, n),
                         "urban_rural": rng.choice(["U", "R"], n)})
    signal = rng.normal(size=n)
    base["wealth_mean"] = signal + (base["urban_rural"] == "U") + rng.normal(scale=0.3, size=n)
    matrices = {"viirs": signal[:, None], "cnn_imagenet": rng.normal(size=(n, 20))}
    for scheme in ["random", "blocks", "kmeans"]:
        table, predictions, groups, folds = evaluate(base, matrices, scheme, CONFIG)
        assert set(table["feature_set"]) == set(BASELINES) | set(matrices)
        assert table[["R2", "R2_ci_low", "R2_ci_high", "R2_fold_std"]].notna().all().all()
        r2 = table.set_index("feature_set")["R2"]
        assert r2["viirs"] > r2["reference_moyenne"]
        assert (folds >= 0).all()
