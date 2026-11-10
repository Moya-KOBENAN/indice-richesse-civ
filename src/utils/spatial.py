"""Groupes spatiaux pour la validation croisée.

Les coordonnées ne servent jamais de variables prédictives : elles définissent
uniquement quelles grappes doivent rester ensemble dans une même partition.
"""
import numpy as np
from sklearn.cluster import KMeans
from sklearn.model_selection import GroupKFold


def to_local_km(latitude, longitude):
    """Projection équirectangulaire locale, suffisante à l'échelle d'un pays."""
    latitude = np.asarray(latitude, dtype=float)
    longitude = np.asarray(longitude, dtype=float)
    x_km = longitude * 111.32 * np.cos(np.radians(latitude.mean()))
    y_km = latitude * 110.57
    return x_km - x_km.min(), y_km - y_km.min()


def regular_blocks(latitude, longitude, block_size_km):
    """Identifiant de bloc carré de `block_size_km` pour chaque point."""
    x_km, y_km = to_local_km(latitude, longitude)
    bx = np.floor(x_km / block_size_km).astype(int)
    by = np.floor(y_km / block_size_km).astype(int)
    return np.array([f"{i}_{j}" for i, j in zip(bx, by)])


def kmeans_groups(latitude, longitude, n_groups, random_state):
    """Groupes géographiques compacts obtenus par KMeans sur les coordonnées."""
    x_km, y_km = to_local_km(latitude, longitude)
    model = KMeans(n_clusters=n_groups, n_init=20, random_state=random_state)
    return model.fit_predict(np.column_stack([x_km, y_km])).astype(str)


def merge_colocated(groups, latitude, longitude):
    """Forcer les grappes aux coordonnées identiques dans le même groupe.

    Sans cela, deux grappes partageant la même image satellite pourraient être
    séparées entre apprentissage et test (fuite d'information).
    """
    groups = np.asarray(groups).astype(object)
    keys = [f"{lat:.6f}_{lon:.6f}" for lat, lon in zip(latitude, longitude)]
    first_group = {}
    for index, key in enumerate(keys):
        groups[index] = first_group.setdefault(key, groups[index])
    return groups.astype(str)


def outer_splits(scheme, frame, config):
    """Partitions externes d'évaluation, identiques pour tous les modèles comparés.

    Renvoie (liste de (train, test), groupes). Les groupes servent aussi à la
    validation interne pour que le choix de alpha respecte la même logique.
    """
    validation = config["validation"]
    seed = config["project"]["random_state"]
    lat, lon = frame["latitude"].to_numpy(), frame["longitude"].to_numpy()
    if scheme == "random":
        # Une grappe = un groupe (sauf coordonnées identiques) : référence non spatiale.
        groups = merge_colocated(np.arange(len(frame)).astype(str), lat, lon)
        splits = shuffled_group_folds(groups, validation["n_splits"], seed)
    else:
        if scheme == "blocks":
            groups = regular_blocks(lat, lon, validation["block_size_km"])
        elif scheme == "kmeans":
            groups = kmeans_groups(lat, lon, validation["kmeans_groups"], seed)
        else:
            raise ValueError(f"Schéma de validation inconnu : {scheme}")
        groups = merge_colocated(groups, lat, lon)
        splitter = GroupKFold(n_splits=validation["n_spatial_folds"])
        splits = list(splitter.split(frame, groups=groups))
    assert_disjoint(splits, groups)
    return splits, groups


def shuffled_group_folds(groups, n_splits, random_state):
    """Plis aléatoires au niveau des groupes (équivalent d'un KFold mélangé)."""
    groups = np.asarray(groups)
    unique = np.random.default_rng(random_state).permutation(np.unique(groups))
    fold_of = {group: position % n_splits for position, group in enumerate(unique)}
    folds = np.array([fold_of[group] for group in groups])
    return [(np.flatnonzero(folds != k), np.flatnonzero(folds == k)) for k in range(n_splits)]


def assert_disjoint(splits, groups):
    """Aucun groupe ne doit apparaître à la fois en apprentissage et en test."""
    groups = np.asarray(groups)
    covered = np.zeros(len(groups), dtype=bool)
    for fold, (train, test) in enumerate(splits, start=1):
        shared = set(groups[train]) & set(groups[test])
        if shared:
            raise ValueError(f"Fuite spatiale au pli {fold} : {sorted(shared)[:5]}")
        covered[test] = True
    if not covered.all():
        raise ValueError("Certaines grappes ne sont dans aucun pli de test.")
