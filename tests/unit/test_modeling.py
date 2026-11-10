"""Ridge imbriquée, références naïves et métriques sur données synthétiques."""
import numpy as np
import pytest

from src.utils.metrics import bootstrap_ci, bootstrap_r2_difference, regression_metrics
from src.utils.modeling import baseline_predictions, fit_final_ridge, nested_ridge_predictions
from src.utils.spatial import shuffled_group_folds

ALPHAS = np.logspace(-2, 3, 8)


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 10))
    y = X[:, 0] * 2 + rng.normal(scale=0.5, size=200)
    groups = np.repeat(np.arange(40), 5).astype(str)
    return X, y, groups, shuffled_group_folds(groups, 5, 0)


def test_nested_ridge_learns_signal(data):
    X, y, groups, splits = data
    pred, folds, alphas = nested_ridge_predictions(X, y, splits, groups, ALPHAS, inner_folds=4)
    assert not np.isnan(pred).any() and set(folds) == {1, 2, 3, 4, 5}
    assert regression_metrics(y, pred)["R2"] > 0.8
    assert ALPHAS[0] <= alphas["alpha_median"] <= ALPHAS[-1]


def test_mean_baseline_has_no_skill(data):
    _, y, _, splits = data
    assert regression_metrics(y, baseline_predictions(y, splits))["R2"] < 0.05


def test_group_baseline_uses_training_means():
    y = np.array([1.0, 1.0, 3.0, 3.0])
    splits = [(np.array([0, 2]), np.array([1, 3])), (np.array([1, 3]), np.array([0, 2]))]
    pred = baseline_predictions(y, splits, np.array(["R", "R", "U", "U"]))
    assert np.allclose(pred, y)


def test_bootstrap_intervals_contain_estimate(data):
    X, y, groups, splits = data
    pred, _, _ = nested_ridge_predictions(X, y, splits, groups, ALPHAS, inner_folds=4)
    ci = bootstrap_ci(y, pred, groups, n_iterations=200)
    assert ci["R2_ci_low"] <= regression_metrics(y, pred)["R2"] <= ci["R2_ci_high"]
    gain = bootstrap_r2_difference(y, pred, np.full_like(y, y.mean()), groups, n_iterations=200)
    assert gain["gain_ci_low"] > 0


def test_final_model_predicts(data):
    X, y, groups, _ = data
    model, alpha = fit_final_ridge(X, y, groups, ALPHAS, inner_folds=4)
    assert model.predict(X[:3]).shape == (3,) and alpha in ALPHAS


def test_residual_ridge_keeps_strong_single_predictor():
    """VIIRS ne doit pas être écrasée par de nombreuses variables bruitées."""
    rng = np.random.default_rng(1)
    n = 300
    viirs = rng.normal(size=n)
    noise = rng.normal(size=(n, 500))
    y = 2 * viirs + rng.normal(scale=0.3, size=n)
    X = np.column_stack([noise, viirs])
    groups = np.arange(n).astype(str)
    splits = shuffled_group_folds(groups, 5, 0)
    plain, _, _ = nested_ridge_predictions(X, y, splits, groups, ALPHAS, 4)
    residual, _, _ = nested_ridge_predictions(X, y, splits, groups, ALPHAS, 4, residual=True)
    assert regression_metrics(y, residual)["R2"] > regression_metrics(y, plain)["R2"]
    assert regression_metrics(y, residual)["R2"] > 0.9
