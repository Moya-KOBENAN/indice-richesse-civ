"""Lecture de config/config.yaml.

Le fichier est lu une seule fois par processus. Les chemins externes
(`~/...`) et relatifs à la racine du projet sont résolus par `resolve_path`.
"""
from functools import lru_cache
from pathlib import Path

import yaml

from src.utils.paths import CONFIG_DIR, PROJECT_ROOT


@lru_cache(maxsize=1)
def load_config():
    with (CONFIG_DIR / "config.yaml").open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_path(value):
    """Chemin absolu : `~` développé, chemin relatif rattaché à la racine."""
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def alpha_grid():
    """Grille de pénalisation Ridge, identique pour toutes les comparaisons."""
    import numpy as np

    spec = load_config()["model"]["alphas"]
    return np.logspace(spec["log10_min"], spec["log10_max"], spec["n"])
