"""Tous les tests de ce dossier lisent les données locales : marqueur `data`.

Exécution rapide sans données : `pytest -m "not data"`.
"""
import pytest


def pytest_collection_modifyitems(items):
    for item in items:
        if "data_quality" in str(item.fspath):
            item.add_marker(pytest.mark.data)
