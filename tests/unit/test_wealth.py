"""Stabilité de l’indice et protection contre les recodages silencieux."""
import numpy as np
import pytest
from src.utils.wealth import weighted_pca_first_component, weighted_quintiles


@pytest.mark.parametrize('weights', [[0., 1., 1.], [-1., 1., 1.], [np.nan, 1., 1.], [1., 1.]])
def test_pca_rejects_invalid_weights(weights):
    with pytest.raises(ValueError):
        weighted_pca_first_component(np.arange(6.).reshape(3, 2), weights)


def test_pca_rejects_constant_design():
    with pytest.raises(ValueError, match='variance positive'):
        weighted_pca_first_component(np.ones((3, 2)), np.ones(3))


def test_pca_invariant_to_weight_rescaling():
    X = np.array([[0., 1.], [1., 2.], [4., 3.]])
    a = weighted_pca_first_component(X, [1., 2., 3.])
    b = weighted_pca_first_component(X, [10., 20., 30.])
    assert np.allclose(a[0], b[0])
    assert np.allclose(a[2], b[2])


def test_quintiles_do_not_split_equal_scores_or_depend_on_row_order():
    scores = np.array([0., 0., 0., 1., 2., 3.])
    weights = np.array([1., 5., 2., 1., 1., 1.])
    q = weighted_quintiles(scores, weights)
    order = np.array([5, 3, 1, 2, 0, 4])
    assert len(set(q[:3])) == 1
    assert np.array_equal(weighted_quintiles(scores[order], weights[order]), q[order])


def test_quintiles_distinct_equal_weight_scores_fill_five_classes():
    assert np.array_equal(weighted_quintiles([1., 2., 3., 4., 5.], np.ones(5)), [1, 2, 3, 4, 5])


def test_dhs_aggregation_rejects_invalid_weights():
    import importlib
    import pandas as pd
    aggregate = importlib.import_module('src.data.dhs.01_prepare_dhs').aggregate_clusters
    frame = pd.DataFrame({'hv001': [1, 1], 'hv005': [1., -1.],
                          'hv270': [1., 2.], 'hv271': [0., 1.]})
    with pytest.raises(ValueError, match='poids'):
        aggregate(frame, 100000)


def test_dhs_target_uses_household_weights_when_they_differ():
    import importlib
    import pandas as pd
    aggregate = importlib.import_module('src.data.dhs.01_prepare_dhs').aggregate_clusters
    frame = pd.DataFrame({'hv001': [1, 1], 'hv005': [1., 3.],
                          'hv270': [1., 2.], 'hv271': [0., 4.]})
    result = aggregate(frame, 1.)
    assert result.wealth_mean.iloc[0] == pytest.approx(3.)
    assert result.wealth_mean_unweighted.iloc[0] == pytest.approx(2.)
