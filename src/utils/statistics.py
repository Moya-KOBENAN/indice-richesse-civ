"""Statistiques descriptives pondérées ; aucun intervalle de sondage implicite."""
import numpy as np
import pandas as pd


def valid_pairs(values, weights):
    x = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(float)
    w = pd.to_numeric(pd.Series(weights), errors="coerce").to_numpy(float)
    if len(x) != len(w):
        raise ValueError("Valeurs et poids de tailles différentes")
    if np.any(~np.isfinite(w) | (w <= 0)):
        raise ValueError("Les poids doivent être finis et strictement positifs")
    mask = np.isfinite(x)
    return x, w, mask


def weighted_quantile(values, weights, probability):
    """Inverse de la CDF empirique pondérée (pas d'interpolation)."""
    if not 0 <= probability <= 1:
        raise ValueError("Probabilité hors de [0, 1]")
    x, w, valid = valid_pairs(values, weights)
    x, w = x[valid], w[valid]
    if not len(x):
        return float("nan")
    order = np.argsort(x, kind="stable")
    x, w = x[order], w[order]
    index = min(np.searchsorted(np.cumsum(w) / w.sum(), probability), len(x) - 1)
    return float(x[index])


def weighted_summary(values, weights):
    x, w, valid = valid_pairs(values, weights)
    result = {"n": len(x), "n_valid": int(valid.sum()), "missing_rate": float((~valid).mean()),
              "weighted_missing_rate": float(w[~valid].sum() / w.sum()) if len(w) else float("nan")}
    xv, wv = x[valid], w[valid]
    if not len(xv):
        return {**result, **dict.fromkeys(["mean", "sd", "min", "q01", "q25", "median", "q75", "q99", "max", "kish_n"], float("nan"))}
    mean = np.average(xv, weights=wv)
    result.update(mean=float(mean), sd=float(np.sqrt(np.average((xv - mean)**2, weights=wv))),
                  min=float(xv.min()), max=float(xv.max()), kish_n=float(wv.sum()**2 / np.sum(wv**2)))
    result.update({label: weighted_quantile(xv, wv, q) for label, q in [("q01", .01), ("q25", .25), ("median", .5), ("q75", .75), ("q99", .99)]})
    return result


def category_frequencies(values, weights):
    """Dénominateur = tous les ménages du groupe, catégorie manquante incluse."""
    weights = np.asarray(weights, dtype=float)
    valid_pairs(np.zeros(len(weights)), weights)
    frame = pd.DataFrame({"category": pd.Series(values).reset_index(drop=True).astype('string').fillna('MANQUANT'), "weight": weights})
    table = frame.groupby("category", sort=True, dropna=False).agg(n=("weight", "size"), weight_sum=("weight", "sum")).reset_index()
    table["share"] = table.n / len(frame)
    table["weighted_share"] = table.weight_sum / table.weight_sum.sum()
    return table


def weighted_correlation(x, y, weights):
    x, w, valid = valid_pairs(x, weights)
    y = pd.to_numeric(pd.Series(y), errors="coerce").to_numpy(float)
    if len(y) != len(x):
        raise ValueError("Variables de tailles différentes")
    valid &= np.isfinite(y)
    x, y, w = x[valid], y[valid], w[valid]
    if len(x) < 2:
        return float("nan")
    x = x - np.average(x, weights=w)
    y = y - np.average(y, weights=w)
    denominator = np.sqrt(np.sum(w*x*x)*np.sum(w*y*y))
    return float(np.sum(w*x*y)/denominator) if denominator > 0 else float("nan")


def missing_profile(frame, columns, weights):
    return pd.DataFrame([{"variable": c, "n_missing": int(frame[c].isna().sum()),
                          "missing_rate": float(frame[c].isna().mean()),
                          "weighted_missing_rate": float(np.average(frame[c].isna(), weights=weights))}
                         for c in columns])
