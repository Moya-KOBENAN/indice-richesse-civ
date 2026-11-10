"""Contrats des couches et import des étapes sans calcul ni données."""
import ast
import os
from pathlib import Path
import subprocess
import sys

from src.pipeline import PIPELINES

ROOT = Path(__file__).resolve().parents[1]


def test_shared_utilities_do_not_depend_on_pipeline_layers():
    for path in (ROOT / 'src/utils').glob('*.py'):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom) and node.module:
                assert not node.module.startswith(('src.data', 'src.models', 'src.features',
                                                   'src.evaluation', 'src.visualization')), path.name


def test_every_registered_stage_imports_without_writes_or_network(tmp_path):
    # Les bibliothèques sont initialisées avant l’audit ; seules les opérations
    # provoquées par l’import des étapes sont surveillées. Les poids CNN ne sont jamais chargés.
    code = r'''
import importlib
import os
import sys
import pandas
import geopandas
import rasterio
import matplotlib
import matplotlib.pyplot
import ee
import sklearn
import scipy.stats
import torch
import torchvision

active = True

def audit(event, args):
    if not active:
        return
    if event == 'open':
        _, mode, flags = args
        if (mode and any(c in mode for c in 'wax+')) or (flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT)):
            raise RuntimeError('Écriture lors de l’import')
    if event in ('os.mkdir', 'os.remove', 'os.rename', 'socket.connect', 'subprocess.Popen'):
        raise RuntimeError('Effet de bord lors de l’import : ' + event)

sys.addaudithook(audit)
for module in sys.argv[1:]:
    loaded = importlib.import_module(module)
    assert callable(loaded.main), module
active = False
'''
    modules = sorted({'src.' + stage.script[:-3].replace('/', '.')
                      for stages in PIPELINES.values() for stage in stages.values()})
    env = {**os.environ, 'PYTHONPATH': str(ROOT), 'PYTHONDONTWRITEBYTECODE': '1',
           'MPLCONFIGDIR': str(tmp_path / 'mpl')}
    result = subprocess.run([sys.executable, '-c', code, *modules], cwd=tmp_path,
                            env=env, capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stderr


def test_models_do_not_depend_on_evaluation_entrypoints():
    for path in (ROOT / 'src/models').rglob('*.py'):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.ImportFrom) and node.module:
                assert not node.module.startswith('src.evaluation'), path.name
