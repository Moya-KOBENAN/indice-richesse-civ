"""Configuration publique et surcharge locale confidentielle facultative.

RICHESSE_CONFIG_FILE désigne un YAML local de surcharge. Les chemins relatifs
restent rattachés à la racine du projet. Aucun fichier local n’est publié.
"""
from copy import deepcopy
from functools import lru_cache
import os
from pathlib import Path

import yaml

from src.utils.paths import CONFIG_DIR, PROJECT_ROOT


def _read_mapping(path):
    with Path(path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("La configuration YAML doit être un dictionnaire.")
    return config


def _merge(base, override):
    result = deepcopy(base)
    for key, value in override.items():
        if key not in result:
            raise ValueError(f"Paramètre de configuration inconnu : {key}")
        if isinstance(result[key], dict):
            if not isinstance(value, dict):
                raise ValueError(f"Le paramètre {key} attend un dictionnaire.")
            result[key] = _merge(result[key], value)
        else:
            result[key] = value
    return result


def read_config(override=None):
    config = _read_mapping(CONFIG_DIR / "config.yaml")
    if override:
        config = _merge(config, _read_mapping(Path(override).expanduser()))
    if os.environ.get("EE_PROJECT_ID"):
        config["earth_engine"]["project_id"] = os.environ["EE_PROJECT_ID"]
    validation = config["validation"]
    for key in ("n_splits", "n_spatial_folds", "inner_folds", "kmeans_groups"):
        if not isinstance(validation[key], int) or validation[key] < 2:
            raise ValueError(f"validation.{key} doit être un entier supérieur ou égal à 2.")
    if validation["block_size_km"] <= 0:
        raise ValueError("La taille des blocs doit être positive.")
    return config


@lru_cache(maxsize=1)
def load_config():
    """Lecture unique par processus ; définir les variables avant le lancement."""
    return read_config(os.environ.get("RICHESSE_CONFIG_FILE"))


def resolve_path(value):
    """Chemin absolu : ~ développé, relatif rattaché à la racine."""
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def alpha_grid():
    import numpy as np

    spec = load_config()["model"]["alphas"]
    return np.logspace(spec["log10_min"], spec["log10_max"], spec["n"])
