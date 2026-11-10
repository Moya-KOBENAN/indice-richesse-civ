"""Catalogue des jeux de variables explicatives.

Un seul endroit définit où se trouve chaque jeu de caractéristiques, pour
chaque enquête et pour la grille nationale. Les étapes d'évaluation et de
prédiction ne manipulent que des noms (« viirs », « cnn_imagenet_tiles »...).
"""
from src.utils.paths import COMMON_PROCESSED_DIR, PROCESSED_DIR

VIIRS_COLUMN = "viirs_2u5r_mean"
# Variantes VIIRS utilisables dans un jeu (« cnn_imagenet_tiles+viirs5 »).
# viirs  : 2 km urbain / 5 km rural (meilleure référence, nécessite le milieu)
# viirs5 : 5 km pour tous (calculable partout, donc sur la grille nationale)
VIIRS_VARIANTS = {"viirs": VIIRS_COLUMN, "viirs5": "viirs_5all_mean"}

IMAGE_SOURCES = {
    # zone : (dossier d'images relatif à data/raw, préfixe du nom, colonne identifiant)
    "dhs": ("satellite", "cluster", "cluster"),
    "ehcvm": ("ehcvm/satellite", "cluster", "cluster"),
    "grid": ("satellite_national", "grid", "grid_id"),
}

# Fichiers produits avant la réorganisation, conservés pour les résultats existants.
_LEGACY_DHS = {
    "cnn_imagenet": "cnn_features_resnet50.csv",
    "cnn_national": "cnn_features_resnet50_viirs_national.csv",
    "cnn_dhs_adapted_exploratory": "cnn_features_resnet50_viirs.csv",
}

CNN_SETS = ("cnn_imagenet", "cnn_imagenet_tiles", "cnn_national", "cnn_national_tiles")


def cnn_file(zone, name):
    """Chemin du fichier de caractéristiques `name` pour `zone` (dhs, ehcvm, grid)."""
    directory = COMMON_PROCESSED_DIR if zone == "grid" else PROCESSED_DIR / zone
    prefix = "grid_" if zone == "grid" else ""
    path = directory / f"{prefix}cnn_features_{name.removeprefix('cnn_')}.csv"
    if zone == "dhs" and not path.exists() and name in _LEGACY_DHS:
        return directory / _LEGACY_DHS[name]
    return path


def parse_feature_set(spec):
    """« cnn_imagenet+viirs » → (['cnn_imagenet'], 'viirs_2u5r_mean') ; sans VIIRS → None."""
    parts = spec.split("+")
    if len(set(parts)) != len(parts):
        raise ValueError(f"Variables répétées dans le jeu : {spec}")
    viirs = [p for p in parts if p in VIIRS_VARIANTS]
    if len(viirs) > 1:
        raise ValueError(f"Une seule variante VIIRS par jeu : {spec}")
    cnn = [p for p in parts if p not in VIIRS_VARIANTS]
    unknown = [p for p in cnn if p not in CNN_SETS and p not in _LEGACY_DHS]
    if unknown:
        raise ValueError(f"Jeu de variables inconnu : {unknown}")
    return cnn, VIIRS_VARIANTS[viirs[0]] if viirs else None


def uses_residual(spec):
    """Combinaison CNN + VIIRS : VIIRS non pénalisée, CNN sur le résidu."""
    cnn, viirs = parse_feature_set(spec)
    return bool(cnn) and viirs is not None
