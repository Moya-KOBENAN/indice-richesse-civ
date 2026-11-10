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
l'ACP ; elle sert seulement de contrôle de cohérence (corrélation de rang non pondérée).
La cible descriptive est fixée sur l’échantillon complet : la validation
prédictive est conditionnelle à cette définition et ne valide pas toute la
procédure de construction de l’indice sur de nouveaux ménages.
"""
from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.data.ehcvm.wealth_index import housing_dummies, select_assets
from src.utils.config import load_config
from src.utils.io import read_table, write_table
from src.utils.paths import EHCVM_INTERIM_DIR, EHCVM_TABLES_DIR, ensure_layout
from src.utils.wealth import positive_weights, weighted_pca_first_component, weighted_quintiles
from src.utils.statistics import weighted_correlation

INPUT_FILE = EHCVM_INTERIM_DIR / "ehcvm_households.csv"
OUTPUT_FILE = EHCVM_INTERIM_DIR / "ehcvm_households_wealth.csv"
LOADINGS_FILE = EHCVM_TABLES_DIR / "wealth_index_loadings.csv"
DIAGNOSTICS_FILE = EHCVM_TABLES_DIR / "wealth_index_diagnostics.csv"


def main():
    ensure_layout("ehcvm")
    params = load_config()["ehcvm"]["wealth_index"]
    households = read_table(INPUT_FILE, required=["grappe", "hhweight", "hhsize", "s11q02"])
    weights = positive_weights(households["hhweight"], len(households))
    if not np.isfinite(households["hhsize"]).all() or (households["hhsize"] <= 0).any():
        raise ValueError("Taille de ménage invalide.")

    assets = select_assets(households, weights, params["min_asset_prevalence"], params["max_asset_prevalence"])
    incomplete = assets.isna().any(axis=1)
    print(f"Ménages exclus pour code d'actif invalide : {int(incomplete.sum())}")
    households, assets, weights = households.loc[~incomplete], assets.loc[~incomplete], weights[~incomplete]

    housing = housing_dummies(households, params["housing_variables"], weights, params["min_category_share"])
    if (households["s11q02"].dropna() <= 0).any():
        raise ValueError("Nombre de pièces non positif : recodage à examiner.")
    rooms = np.log(households["s11q02"] / households["hhsize"])
    if not rooms.notna().any():
        raise ValueError("Aucun nombre de pièces observé.")
    rooms = rooms.fillna(np.average(rooms.dropna(), weights=weights[rooms.notna()]))
    design = pd.concat([housing, assets, rooms.rename("log_pieces_par_personne")], axis=1)

    score, loading, explained, kept = weighted_pca_first_component(design.to_numpy(float), weights)
    n_assets = assets.sum(axis=1).to_numpy()
    if weighted_correlation(score, n_assets, weights) < 0:
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
        {"indicateur": "variance_expliquee_axe2", "valeur": explained[1] if len(explained) > 1 else np.nan},
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
