"""Métriques de performance et incertitude.

Un score seul ne suffit pas : on rapporte aussi sa variabilité entre plis
et un intervalle bootstrap par groupes spatiaux, pour savoir si l'écart entre
deux modèles dépasse le bruit d'échantillonnage.
"""
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "R2": r2_score(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": mean_absolute_error(y_true, y_pred),
        "Spearman": float(spearmanr(y_true, y_pred).statistic),
        "n": len(y_true),
    }


def fold_dispersion(y_true, y_pred, folds):
    """Moyenne et écart-type du R² calculé pli par pli."""
    y_true, y_pred, folds = map(np.asarray, (y_true, y_pred, folds))
    scores = [r2_score(y_true[folds == k], y_pred[folds == k])
              for k in np.unique(folds) if (folds == k).sum() > 1]
    return {"R2_fold_mean": float(np.mean(scores)), "R2_fold_std": float(np.std(scores, ddof=1))}


def _group_bootstrap_indices(groups, n_iterations, random_state):
    """Tirages avec remise des groupes spatiaux (pas des grappes isolées).

    Rééchantillonner les groupes respecte l'autocorrélation spatiale ; tirer
    les grappes une à une donnerait des intervalles trop étroits.
    """
    groups = np.asarray(groups)
    members = {g: np.flatnonzero(groups == g) for g in np.unique(groups)}
    labels = list(members)
    rng = np.random.default_rng(random_state)
    for _ in range(n_iterations):
        drawn = rng.choice(len(labels), size=len(labels), replace=True)
        yield np.concatenate([members[labels[i]] for i in drawn])


def bootstrap_ci(y_true, y_pred, groups, n_iterations=1000, random_state=0, level=0.95):
    """Intervalles bootstrap par groupes pour R² et RMSE."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    r2, rmse = [], []
    for index in _group_bootstrap_indices(groups, n_iterations, random_state):
        r2.append(r2_score(y_true[index], y_pred[index]))
        rmse.append(np.sqrt(mean_squared_error(y_true[index], y_pred[index])))
    low, high = (1 - level) / 2 * 100, (1 + level) / 2 * 100
    return {"R2_ci_low": np.percentile(r2, low), "R2_ci_high": np.percentile(r2, high),
            "RMSE_ci_low": np.percentile(rmse, low), "RMSE_ci_high": np.percentile(rmse, high)}


def bootstrap_r2_difference(y_true, pred_a, pred_b, groups, n_iterations=1000, random_state=0, level=0.95):
    """Intervalle du gain R²(a) − R²(b) sur les mêmes tirages (comparaison appariée)."""
    y_true, pred_a, pred_b = (np.asarray(v, float) for v in (y_true, pred_a, pred_b))
    gains = [r2_score(y_true[i], pred_a[i]) - r2_score(y_true[i], pred_b[i])
             for i in _group_bootstrap_indices(groups, n_iterations, random_state)]
    low, high = (1 - level) / 2 * 100, (1 + level) / 2 * 100
    return {"gain_R2": r2_score(y_true, pred_a) - r2_score(y_true, pred_b),
            "gain_ci_low": np.percentile(gains, low), "gain_ci_high": np.percentile(gains, high)}


def subgroup_metrics(y_true, y_pred, labels):
    """Performance par sous-population (milieu, région) : l'erreur est-elle homogène ?"""
    y_true, y_pred, labels = map(np.asarray, (y_true, y_pred, labels))
    rows = []
    for label in np.unique(labels):
        mask = labels == label
        if mask.sum() >= 3:
            rows.append({"subgroup": label, **regression_metrics(y_true[mask], y_pred[mask])})
    return rows
