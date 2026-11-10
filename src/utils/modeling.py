"""Régression Ridge avec validation croisée imbriquée.

Protocole commun à toutes les comparaisons :
- plis externes fournis par l'appelant (aléatoires ou spatiaux) ;
- choix de alpha par GroupKFold interne sur les seuls groupes d'apprentissage,
  pour que la sélection imite la situation évaluée (extrapolation spatiale) ;
- imputation et standardisation apprises dans chaque pli (pas de fuite).
"""
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge

from src.utils.spatial import assert_disjoint


def ridge_pipeline():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("ridge", Ridge()),
    ])


class ResidualRidge(BaseEstimator, RegressorMixin):
    """VIIRS (dernière colonne) par MCO, puis Ridge des CNN sur le résidu.

    Concaténer une variable très informative (VIIRS) à 2 048 caractéristiques
    CNN sous une même pénalité l'écrase : avec alpha ≈ 10³, son coefficient
    est rétréci comme les autres. Ici VIIRS n'est pas pénalisée et le CNN
    n'explique que ce que VIIRS n'explique pas : cette variante évalue une stratégie séquentielle particulière.
    Elle ne remplace pas toutes les formes de régression conjointe CNN + VIIRS.
    """

    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y, float)
        self.base_ = LinearRegression().fit(X[:, -1:], y)
        residual = y - self.base_.predict(X[:, -1:])
        self.ridge_ = ridge_pipeline().set_params(ridge__alpha=self.alpha).fit(X[:, :-1], residual)
        return self

    def predict(self, X):
        X = np.asarray(X, float)
        return self.base_.predict(X[:, -1:]) + self.ridge_.predict(X[:, :-1])


def _inner_search(alphas, groups_train, inner_folds, residual=False):
    n_folds = min(inner_folds, len(np.unique(groups_train)))
    if n_folds < 2:
        raise ValueError("Pas assez de groupes d'apprentissage pour choisir alpha.")
    estimator, grid = (ResidualRidge(), {"alpha": alphas}) if residual \
        else (ridge_pipeline(), {"ridge__alpha": alphas})
    return GridSearchCV(estimator, grid, scoring="neg_root_mean_squared_error",
                        cv=GroupKFold(n_splits=n_folds), n_jobs=1)


def _best_alpha(search):
    params = search.best_params_
    return float(params.get("alpha", params.get("ridge__alpha")))


def nested_ridge_predictions(X, y, splits, groups, alphas, inner_folds=5, residual=False):
    """Prédictions hors échantillon et alpha retenu dans chaque pli externe."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    groups = np.asarray(groups)
    splits = list(splits)
    if X.ndim != 2 or y.ndim != 1 or len(X) != len(y) or len(groups) != len(y):
        raise ValueError("Dimensions incompatibles entre X, y et groupes.")
    if not np.isfinite(y).all():
        raise ValueError("La cible doit être finie.")
    assert_disjoint(splits, groups)
    predictions = np.full(len(y), np.nan)
    folds = np.full(len(y), -1)
    chosen = []
    for fold, (train, test) in enumerate(splits, start=1):
        search = _inner_search(alphas, groups[train], inner_folds, residual)
        search.fit(X[train], y[train], groups=groups[train])
        predictions[test] = search.predict(X[test])
        folds[test] = fold
        chosen.append(_best_alpha(search))
    if np.isnan(predictions).any():
        raise ValueError("Des grappes n'ont reçu aucune prédiction.")
    alpha_at_bound = any(a in (alphas[0], alphas[-1]) for a in chosen)
    return predictions, folds, {"alpha_median": float(np.median(chosen)),
                                "alpha_min": min(chosen), "alpha_max": max(chosen),
                                "alpha_at_grid_bound": alpha_at_bound}


def baseline_predictions(y, splits, labels=None):
    """Référence naïve : moyenne d'apprentissage, globale ou par modalité.

    Un modèle satellitaire n'a d'intérêt que s'il fait mieux que ces règles
    simples (par ex. « urbain = plus riche »).
    """
    y = np.asarray(y, dtype=float)
    predictions = np.full(len(y), np.nan)
    for train, test in splits:
        overall = y[train].mean()
        if labels is None:
            predictions[test] = overall
            continue
        labels = np.asarray(labels)
        means = {lab: y[train][labels[train] == lab].mean() for lab in np.unique(labels[train])}
        predictions[test] = [means.get(lab, overall) for lab in labels[test]]
    return predictions


def fit_final_ridge(X, y, groups, alphas, inner_folds=5, residual=False):
    """Modèle final sur toutes les grappes, alpha choisi par validation spatiale."""
    search = _inner_search(alphas, np.asarray(groups), inner_folds, residual)
    search.fit(np.asarray(X, float), np.asarray(y, float), groups=np.asarray(groups))
    # refit=True : best_estimator_ est déjà réentraîné sur toutes les grappes.
    return search.best_estimator_, _best_alpha(search)
