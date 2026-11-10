"""Assemblage partagé des matrices explicatives, sans dépendance à l’évaluation."""
import numpy as np
import pandas as pd

from src.utils.feature_sets import VIIRS_VARIANTS, cnn_file, parse_feature_set
from src.utils.io import read_table
from src.utils.paths import survey_clusters_file


def feature_columns(frame, path):
    """Colonnes de caractéristiques : `cnn_*` (nouvelles étapes) ou `feature_*` (étape 13)."""
    features = frame.filter(regex=r"^(cnn|feature)_\d+$")
    if features.shape[1] == 0:
        raise ValueError(f"{path.name} : aucune colonne de caractéristiques reconnue.")
    columns = sorted(features.columns, key=lambda name: int(name.rsplit("_", 1)[1]))
    return features[columns]


def load_design(survey, feature_sets):
    """Table des grappes + matrices X par jeu disponible, sur des grappes communes."""
    if not feature_sets:
        raise ValueError("Aucun jeu de variables demandé.")
    requested_viirs = {parse_feature_set(spec)[1] for spec in feature_sets} - {None}
    base = read_table(survey_clusters_file(survey),
                      required=["cluster", "wealth_mean", "latitude", "longitude", "urban_rural", *sorted(requested_viirs)],
                      unique="cluster")
    if not np.isfinite(base[["wealth_mean", "latitude", "longitude"]].to_numpy(float)).all():
        raise ValueError("Cible ou coordonnées non finies.")
    if not base.latitude.between(-90, 90).all() or not base.longitude.between(-180, 180).all():
        raise ValueError("Coordonnées hors limites.")
    if not base.urban_rural.isin(["U", "R"]).all():
        raise ValueError("Milieu inconnu.")
    blocks, available = {}, []
    for spec in feature_sets:
        cnn_names, _ = parse_feature_set(spec)
        paths = [cnn_file(survey, name) for name in cnn_names]
        missing = [p.name for p in paths if not p.exists()]
        if missing:
            print(f"Jeu « {spec} » ignoré : fichiers absents {missing}")
            continue
        for name, path in zip(cnn_names, paths):
            if name not in blocks:
                frame = read_table(path, required=["cluster"], unique="cluster")
                blocks[name] = feature_columns(frame.set_index("cluster"), path).add_prefix(f"{name}__")
        available.append(spec)
    if not available:
        raise ValueError("Aucun jeu de variables disponible : exécuter les extractions requises.")
    clusters = set(base["cluster"])
    for frame in blocks.values():
        clusters &= set(frame.index)
    if not clusters:
        raise ValueError("Aucune grappe commune aux jeux de caractéristiques.")
    dropped = len(base) - len(clusters)
    if dropped:
        print(f"{dropped} grappes sans toutes les caractéristiques : exclues pour une comparaison équitable.")
    base = base[base["cluster"].isin(clusters)].sort_values("cluster").reset_index(drop=True)
    matrices = {}
    for spec in available:
        cnn_names, viirs_column = parse_feature_set(spec)
        parts = [blocks[n].loc[base["cluster"]].to_numpy() for n in cnn_names]
        if viirs_column:
            # VIIRS toujours en dernière colonne (attendu par ResidualRidge)
            parts.append(np.log1p(base[[viirs_column]].clip(lower=0)).to_numpy())
        matrices[spec] = np.hstack(parts)
    return base, matrices
