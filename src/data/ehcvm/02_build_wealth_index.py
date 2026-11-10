"""EHCVM 02 — Indice de richesse des ménages par ACP pondérée.

Entrée  : data/interim/ehcvm/ehcvm_households.csv (étape 01)
Sorties : data/interim/ehcvm/ehcvm_households_wealth.csv
          outputs/ehcvm/tables/wealth_index_loadings.csv
          outputs/ehcvm/tables/wealth_index_diagnostics.csv

Méthode (proche de l'indice DHS, Rutstein & Johnson 2004) :
- logement : une indicatrice par modalité (aucun libellé inventé, aucun
  code traité comme un nombre) ; modalités rares regroupées en « autre » ;
  valeur manquante = modalité « manquant » explicite ;
- biens : possession 0/1 ; biens quasi universels ou quasi absents exclus ;
- densité : pièces par personne (log) ;
- standardisation et covariance pondérées par hhweight ;
- score = première composante, orientée pour croître avec le nombre de biens.

Statut de la cible : l'indice est construit une fois sur tous les ménages,
sans aucune information satellitaire, comme l'indice DHS. Il ne dépend donc
pas des partitions de validation. La consommation (pcexp) n'entre pas dans
l'ACP ; elle sert seulement de validation externe (corrélation de rang).
"""
from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.utils.config import load_config
from src.utils.io import read_table, write_table
from src.utils.paths import EHCVM_INTERIM_DIR, EHCVM_TABLES_DIR, ensure_layout

INPUT_FILE = EHCVM_INTERIM_DIR / "ehcvm_households.csv"
OUTPUT_FILE = EHCVM_INTERIM_DIR / "ehcvm_households_wealth.csv"
LOADINGS_FILE = EHCVM_TABLES_DIR / "wealth_index_loadings.csv"
DIAGNOSTICS_FILE = EHCVM_TABLES_DIR / "wealth_index_diagnostics.csv"


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
    prevalence = assets.apply(lambda col: np.average(col.fillna(0), weights=weights))
    kept = prevalence[(prevalence >= low) & (prevalence <= high)].index
    print(f"Biens retenus : {len(kept)}/{assets.shape[1]} "
          f"(exclus : {sorted(set(assets.columns) - set(kept))})")
    return assets[kept]


def weighted_pca_first_component(X, weights):
    """Première composante d'une ACP sur variables standardisées pondérées."""
    w = weights / weights.sum()
    mean = w @ X
    std = np.sqrt(w @ (X - mean) ** 2)
    keep = std > 0
    Z = (X[:, keep] - mean[keep]) / std[keep]
    covariance = (Z * w[:, None]).T @ Z
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
    return Z @ eigenvectors[:, 0], eigenvectors[:, 0], eigenvalues / eigenvalues.sum(), keep


def weighted_quintiles(score, weights):
    order = np.argsort(score)
    cumulative = np.cumsum(weights[order]) / weights.sum()
    quintiles = np.empty(len(score), dtype=int)
    quintiles[order] = np.minimum((cumulative * 5).astype(int) + 1, 5)
    return quintiles


def main():
    ensure_layout("ehcvm")
    params = load_config()["ehcvm"]["wealth_index"]
    households = read_table(INPUT_FILE, required=["grappe", "hhweight", "hhsize", "s11q02"])
    weights = households["hhweight"].to_numpy(float)

    assets = select_assets(households, weights, params["min_asset_prevalence"], params["max_asset_prevalence"])
    incomplete = assets.isna().any(axis=1)
    print(f"Ménages exclus pour code d'actif invalide : {int(incomplete.sum())}")
    households, assets, weights = households.loc[~incomplete], assets.loc[~incomplete], weights[~incomplete]

    housing = housing_dummies(households, params["housing_variables"], weights, params["min_category_share"])
    rooms = np.log(households["s11q02"].clip(lower=1) / households["hhsize"])
    rooms = rooms.fillna(np.average(rooms.dropna(), weights=weights[rooms.notna()]))
    design = pd.concat([housing, assets, rooms.rename("log_pieces_par_personne")], axis=1)

    score, loading, explained, kept = weighted_pca_first_component(design.to_numpy(float), weights)
    n_assets = assets.sum(axis=1).to_numpy()
    if np.corrcoef(score, n_assets)[0, 1] < 0:
        score, loading = -score, -loading

    result = households[["grappe", "menage", "vague", "hhweight", "hhsize", "milieu",
                         "region", "pcexp", "latitude", "longitude"]].copy()
    result["wealth_score"] = score
    result["wealth_quintile"] = weighted_quintiles(score, weights)
    write_table(result, OUTPUT_FILE)

    loadings = pd.DataFrame({"variable": design.columns[kept], "loading": loading})
    write_table(loadings.sort_values("loading", ascending=False), LOADINGS_FILE)
    rho = spearmanr(score, households["pcexp"]).statistic
    diagnostics = pd.DataFrame([
        {"indicateur": "variance_expliquee_axe1", "valeur": explained[0]},
        {"indicateur": "variance_expliquee_axe2", "valeur": explained[1]},
        {"indicateur": "spearman_score_pcexp", "valeur": rho},
        {"indicateur": "n_menages", "valeur": len(result)},
        {"indicateur": "n_variables", "valeur": int(kept.sum())},
    ] + [{"indicateur": f"spearman_score_pcexp_milieu_{m}",
          "valeur": spearmanr(score[mask], households.loc[mask, "pcexp"]).statistic}
         for m in sorted(households["milieu"].unique())
         for mask in [(households["milieu"] == m).to_numpy()]])
    write_table(diagnostics, DIAGNOSTICS_FILE)
    print(diagnostics.to_string(index=False))


if __name__ == "__main__":
    main()
