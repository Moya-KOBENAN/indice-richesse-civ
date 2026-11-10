"""Configuration locale et clés de jointure sûres."""
import pytest
from src.utils.config import read_config
from src.utils.io import read_table


def test_local_override_merges_without_removing_scientific_defaults(tmp_path):
    path = tmp_path / 'local.yaml'
    path.write_text('dhs:\n  raw_dir: data/raw/authorized\n')
    config = read_config(path)
    assert config['dhs']['raw_dir'] == 'data/raw/authorized'
    assert config['dhs']['wealth_score_divisor'] == 100000


def test_local_override_rejects_typo(tmp_path):
    path = tmp_path / 'local.yaml'
    path.write_text('validation:\n  inner_fold: 4\n')
    with pytest.raises(ValueError, match='inconnu'):
        read_config(path)


def test_environment_earth_engine_project_takes_precedence(monkeypatch):
    monkeypatch.setenv('EE_PROJECT_ID', 'test-project')
    assert read_config()['earth_engine']['project_id'] == 'test-project'


def test_table_rejects_missing_unique_key(tmp_path):
    path = tmp_path / 'table.csv'
    path.write_text('id,value\n1,4\n,5\n')
    with pytest.raises(ValueError, match='manquante'):
        read_table(path, unique='id')
