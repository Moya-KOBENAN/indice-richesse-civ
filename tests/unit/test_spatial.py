"""Partitions spatiales : aucune fuite entre apprentissage et test."""
import numpy as np
import pandas as pd
import pytest

from src.utils.spatial import (assert_disjoint, merge_colocated, outer_splits,
                               regular_blocks, shuffled_group_folds)

CONFIG = {"project": {"random_state": 0},
          "validation": {"n_splits": 5, "n_spatial_folds": 4, "block_size_km": 100, "kmeans_groups": 6}}


@pytest.fixture
def clusters():
    rng = np.random.default_rng(1)
    frame = pd.DataFrame({"latitude": rng.uniform(4.5, 10.5, 120), "longitude": rng.uniform(-8.5, -2.5, 120)})
    frame.loc[1, ["latitude", "longitude"]] = frame.loc[0, ["latitude", "longitude"]].to_numpy()
    return frame


def test_blocks_group_nearby_points():
    blocks = regular_blocks([5.0, 5.01, 9.0], [-4.0, -4.01, -7.0], 100)
    assert blocks[0] == blocks[1] != blocks[2]


def test_colocated_clusters_share_group():
    groups = merge_colocated(["a", "b", "c"], [5.0, 5.0, 6.0], [-4.0, -4.0, -5.0])
    assert groups[0] == groups[1] != groups[2]


@pytest.mark.parametrize("scheme", ["random", "blocks", "kmeans"])
def test_outer_splits_cover_all_without_leakage(clusters, scheme):
    splits, groups = outer_splits(scheme, clusters, CONFIG)
    assert_disjoint(splits, groups)
    tested = np.concatenate([test for _, test in splits])
    assert sorted(tested) == list(range(len(clusters)))
    # Les deux grappes au même endroit ne sont jamais séparées
    for train, test in splits:
        assert (0 in test) == (1 in test)


def test_leakage_is_detected():
    with pytest.raises(ValueError, match="Fuite"):
        assert_disjoint([(np.array([0]), np.array([1]))], np.array(["a", "a"]))


def test_shuffled_group_folds_balanced():
    folds = shuffled_group_folds(np.arange(100).astype(str), 5, 0)
    assert [len(test) for _, test in folds] == [20] * 5


def test_distance_diagnostic_detects_close_train_test_points():
    from src.evaluation.common.audit_spatial_splits import distance_diagnostics
    frame = pd.DataFrame({'latitude': [5., 5.001, 9.], 'longitude': [-4., -4., -4.]})
    rows = distance_diagnostics(frame, [(np.array([0, 2]), np.array([1]))], 5.)
    assert rows.nearest_train_min_km.iloc[0] < 1.
    assert rows.share_below_image_side.iloc[0] == 1.
