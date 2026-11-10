"""Recodages patrimoniaux partagés par construction et sensibilité EHCVM."""
import numpy as np
import pandas as pd


def weighted_share(values, weights):
    return (pd.Series(weights, index=values.index).groupby(values).sum() / weights.sum())


def housing_dummies(frame, variables, weights, min_share):
    """Indicatrices par modalité, modalités rares regroupées."""
    blocks = []
    for variable in variables:
        codes = frame[variable].astype("Int64").astype(str).replace("<NA>", "manquant")
        shares = weighted_share(codes, weights)
        rare = shares[shares < min_share].index
        codes = codes.where(~codes.isin(rare), "autre")
        blocks.append(pd.get_dummies(codes, prefix=variable, dtype=float))
    return pd.concat(blocks, axis=1)


def select_assets(frame, weights, low, high):
    assets = frame.filter(like="asset_")
    if assets.empty:
        raise ValueError("Aucune variable d’actif disponible.")
    if not assets.stack().isin([0., 1.]).all():
        raise ValueError("Les actifs doivent être binaires ou manquants.")
    def prevalence_observed(col):
        valid = col.notna().to_numpy()
        return np.average(col.to_numpy()[valid], weights=weights[valid]) if valid.any() else np.nan
    prevalence = assets.apply(prevalence_observed)
    kept = prevalence[(prevalence >= low) & (prevalence <= high)].index
    print(f"Biens retenus : {len(kept)}/{assets.shape[1]} "
          f"(exclus : {sorted(set(assets.columns) - set(kept))})")
    if not len(kept):
        raise ValueError("Aucun actif retenu aux seuils configurés.")
    return assets[kept]
