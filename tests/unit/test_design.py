"""Contrat de l’assemblage des matrices sur des fichiers synthétiques."""
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from src.features import design


def test_feature_columns_sorted_by_numeric_index():
    frame = pd.DataFrame({'cnn_10': [10.], 'cnn_2': [2.], 'cluster': [1]})
    assert list(design.feature_columns(frame, Path('synthetic.csv'))) == ['cnn_2', 'cnn_10']


def test_viirs_design_requires_only_requested_variant(tmp_path, monkeypatch):
    path = tmp_path / 'survey.csv'
    pd.DataFrame({'cluster': [2, 1], 'wealth_mean': [1., 0.],
                  'latitude': [5., 6.], 'longitude': [-4., -5.],
                  'urban_rural': ['U', 'R'], 'viirs_5all_mean': [3., 2.]}).to_csv(path, index=False)
    monkeypatch.setattr(design, 'survey_clusters_file', lambda _: path)
    base, X = design.load_design('dhs', ['viirs5'])
    assert base.cluster.tolist() == [1, 2]
    assert np.allclose(X['viirs5'][:, 0], np.log1p([2., 3.]))


def test_empty_feature_request_rejected():
    with pytest.raises(ValueError, match='Aucun jeu'):
        design.load_design('dhs', [])
