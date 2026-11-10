"""Détecter des évaluations invalides avant tout ajustement."""
import numpy as np
import pytest

from src.utils.spatial import assert_disjoint, shuffled_group_folds
from src.utils.modeling import nested_ridge_predictions
from src.evaluation.common.cross_survey_validation import zscore


@pytest.mark.parametrize('splits', [
    [([0, 1], [1, 2]), ([1, 2], [0])],
    [([1, 2], [0]), ([0, 2], [1]), ([1, 2], [0])],
    [([1, 2], [0])],
    [([1, 2], [0, 0]), ([0], [1, 2])],
    [([1, 2], [-1]), ([0], [1, 2])],
    [([], [0, 1, 2])],
])
def test_rejects_leakage_duplicate_missing_or_invalid_indices(splits):
    with pytest.raises(ValueError):
        assert_disjoint(splits, np.array(['a', 'b', 'c']))


def test_nested_ridge_rejects_shared_geographic_groups():
    X = np.arange(12.).reshape(6, 2)
    groups = np.array(['a', 'b', 'c', 'a', 'b', 'c'])
    with pytest.raises(ValueError, match='Fuite spatiale'):
        nested_ridge_predictions(X, np.arange(6.), [([0, 1, 2], [3, 4, 5])],
                                 groups, [0.1, 1.])


def test_random_folds_reject_too_few_groups():
    with pytest.raises(ValueError):
        shuffled_group_folds(['a', 'b'], 3, 42)


@pytest.mark.parametrize('values', [[1., 1.], [1., np.nan], [1., np.inf]])
def test_standardization_rejects_undefined_comparisons(values):
    with pytest.raises(ValueError):
        zscore(values)


def test_standardization_uses_consistent_population_scale():
    result = zscore([2., 4., 6.])
    assert result.mean() == pytest.approx(0.)
    assert result.std(ddof=0) == pytest.approx(1.)
