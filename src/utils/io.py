"""Lecture et écriture de tables avec contrôles explicites.

Principe : une étape échoue tôt et clairement (colonne absente, clé dupliquée,
fusion incomplète) plutôt que de produire silencieusement une table fausse.
"""
from pathlib import Path

import pandas as pd


def read_table(path, required=(), unique=None):
    """Lire un CSV et vérifier les colonnes obligatoires et l'unicité d'une clé."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Entrée absente : {path}. Exécuter l'étape qui la produit.")
    frame = pd.read_csv(path, low_memory=False)
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{path.name} : colonnes manquantes {missing}")
    if unique is not None and frame.duplicated(unique).any():
        raise ValueError(f"{path.name} : clé {unique} non unique")
    return frame


def write_table(frame, path):
    """Écrire un CSV en créant le dossier ; renvoie le chemin pour les journaux."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    print(f"Écrit : {path} ({len(frame)} lignes, {frame.shape[1]} colonnes)")
    return path


def check_full_merge(merged, indicator="_merge", label="fusion"):
    """Refuser une fusion qui laisse des lignes sans correspondance."""
    counts = merged[indicator].value_counts()
    unmatched = int(counts.get("left_only", 0) + counts.get("right_only", 0))
    if unmatched:
        raise ValueError(f"{label} : {unmatched} lignes sans correspondance ({counts.to_dict()})")
    return merged.drop(columns=indicator)
