"""Opérations numériques de l’indice patrimonial, indépendantes des fichiers."""
import numpy as np

from src.utils.statistics import weighted_quantile


def positive_weights(weights, n):
    weights = np.asarray(weights, dtype=float)
    if weights.shape != (n,) or n == 0 or not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError("Les poids doivent être finis, strictement positifs et alignés.")
    return weights


def weighted_pca_first_component(X, weights):
    """ACP de corrélation pondérée ; les colonnes constantes sont retirées."""
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[0] < 2 or not np.isfinite(X).all():
        raise ValueError("L’ACP exige une matrice finie et au moins deux observations.")
    w = positive_weights(weights, len(X))
    w = w / w.sum()
    mean = w @ X
    std = np.sqrt(w @ (X - mean) ** 2)
    keep = std > np.finfo(float).eps
    if not keep.any():
        raise ValueError("Aucune variable de variance positive pour l’ACP.")
    Z = (X[:, keep] - mean[keep]) / std[keep]
    covariance = (Z * w[:, None]).T @ Z
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.)
    loading = eigenvectors[:, order[0]]
    return Z @ loading, loading, eigenvalues / eigenvalues.sum(), keep


def weighted_quintiles(score, weights):
    """Seuils de la CDF pondérée ; les ex æquo restent dans la même classe.

    En présence d’ex æquo, certaines classes peuvent être vides et leurs poids
    ne sont pas nécessairement égaux. Aucun rang arbitraire ne sépare les scores identiques.
    """
    score = np.asarray(score, dtype=float)
    if score.ndim != 1 or not np.isfinite(score).all():
        raise ValueError("Les scores doivent être un vecteur fini.")
    weights = positive_weights(weights, len(score))
    cuts = [weighted_quantile(score, weights, q) for q in (.2, .4, .6, .8)]
    return np.searchsorted(cuts, score, side="left") + 1
