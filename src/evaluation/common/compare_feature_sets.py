"""Tableau d'ablation : apport de chaque source d'information.

Question : les caractéristiques satellitaires apportent-elles une information
au-delà des lumières nocturnes (H2) et de règles simples ?

Protocole (identique pour toutes les lignes du tableau) :
- mêmes grappes (intersection des jeux disponibles) et mêmes plis externes ;
- trois schémas : aléatoire (référence optimiste), blocs réguliers de 100 km
  et groupes KMeans (extrapolation spatiale, critère principal) ;
- Ridge imbriquée, alpha choisi par validation interne spatiale ; pour les
  combinaisons CNN + VIIRS, VIIRS est estimée sans pénalité et la Ridge CNN
  porte sur le résidu (sinon VIIRS est écrasée parmi 2 048 variables) ;
- références naïves : moyenne globale et moyenne urbain/rural ;
- incertitude : écart-type entre plis, IC bootstrap par groupes spatiaux,
  gain apparié par rapport à VIIRS avec règle de décision consignée
  (config.yaml : validation.min_r2_gain).

Sorties (outputs/<enquête>/tables/) : ablation_metrics.csv,
ablation_gain_vs_viirs.csv, ablation_subgroups.csv, ablation_predictions.csv
"""
from pathlib import Path
import argparse
import sys

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd

from src.utils.config import alpha_grid, load_config
from src.utils.feature_sets import VIIRS_VARIANTS, cnn_file, parse_feature_set, uses_residual
from src.utils.io import read_table, write_table
from src.utils.metrics import (bootstrap_ci, bootstrap_r2_difference, fold_dispersion,
                               regression_metrics, subgroup_metrics)
from src.utils.modeling import baseline_predictions, nested_ridge_predictions
from src.utils.paths import SURVEYS, survey_clusters_file, survey_dirs
from src.utils.spatial import outer_splits

DEFAULT_SETS = ["viirs", "viirs5", "cnn_imagenet", "cnn_imagenet_tiles", "cnn_national",
                "cnn_national_tiles", "cnn_imagenet+viirs", "cnn_imagenet_tiles+viirs",
                "cnn_national_tiles+viirs", "cnn_imagenet_tiles+viirs5"]
BASELINES = ["reference_moyenne", "reference_milieu"]
SCHEMES = ["random", "blocks", "kmeans"]


def feature_columns(frame, path):
    """Colonnes de caractéristiques : `cnn_*` (nouvelles étapes) ou `feature_*` (étape 13)."""
    features = frame.filter(regex=r"^(cnn|feature)_\d+$")
    if features.shape[1] == 0:
        raise ValueError(f"{path.name} : aucune colonne de caractéristiques reconnue.")
    return features


def load_design(survey, feature_sets):
    """Table des grappes + matrices X par jeu disponible, sur des grappes communes."""
    base = read_table(survey_clusters_file(survey),
                      required=["cluster", "wealth_mean", "latitude", "longitude", "urban_rural", *VIIRS_VARIANTS.values()],
                      unique="cluster")
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
    clusters = set(base["cluster"])
    for frame in blocks.values():
        clusters &= set(frame.index)
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


def evaluate(base, matrices, scheme, config):
    splits, groups = outer_splits(scheme, base, config)
    y = base["wealth_mean"].to_numpy(float)
    validation = config["validation"]
    seed = config["project"]["random_state"]
    predictions, rows = {}, []
    for name in BASELINES:
        labels = base["urban_rural"].to_numpy() if name == "reference_milieu" else None
        predictions[name] = (baseline_predictions(y, splits, labels), {}, 0)
    for spec, X in matrices.items():
        # CNN + VIIRS : VIIRS non pénalisée, CNN sur le résidu (voir ResidualRidge).
        residual = uses_residual(spec)
        pred, folds, alphas = nested_ridge_predictions(X, y, splits, groups, alpha_grid(),
                                                       validation["inner_folds"], residual)
        predictions[spec] = (pred, alphas, X.shape[1])
        print(f"  {scheme:7s} {spec:28s} R² = {regression_metrics(y, pred)['R2']:.3f}")
    fold_of = np.empty(len(y), dtype=int)
    for k, (_, test) in enumerate(splits):
        fold_of[test] = k
    for name, (pred, alphas, n_predictors) in predictions.items():
        rows.append({"scheme": scheme, "feature_set": name, "n_predictors": n_predictors,
                     **regression_metrics(y, pred), **fold_dispersion(y, pred, fold_of),
                     **bootstrap_ci(y, pred, groups, validation["bootstrap_iterations"], seed), **alphas})
    return pd.DataFrame(rows), predictions, groups, fold_of


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--survey", choices=SURVEYS, default="dhs")
    parser.add_argument("--sets", nargs="+", default=DEFAULT_SETS)
    parser.add_argument("--schemes", nargs="+", choices=SCHEMES, default=SCHEMES)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    config = load_config()
    base, matrices = load_design(args.survey, args.sets)
    print(f"{len(base)} grappes ; jeux évalués : {list(matrices)}")
    if args.dry_run:
        return
    tables = survey_dirs(args.survey)["tables"]
    y = base["wealth_mean"].to_numpy(float)
    metrics, gains, subgroups, stacked = [], [], [], []
    for scheme in args.schemes:
        table, predictions, groups, folds = evaluate(base, matrices, scheme, config)
        metrics.append(table)
        frame = base[["cluster", "urban_rural"]].assign(scheme=scheme, fold=folds, observed=y)
        for name, (pred, _, _) in predictions.items():
            frame[name] = pred
            subgroups += [{"scheme": scheme, "feature_set": name, **row}
                          for row in subgroup_metrics(y, pred, base["urban_rural"])]
            if "viirs" in predictions and name != "viirs":
                gain = bootstrap_r2_difference(y, pred, predictions["viirs"][0], groups,
                                               config["validation"]["bootstrap_iterations"],
                                               config["project"]["random_state"])
                gain["meilleur_que_viirs"] = (gain["gain_R2"] >= config["validation"]["min_r2_gain"]
                                              and gain["gain_ci_low"] > 0)
                gains.append({"scheme": scheme, "feature_set": name, **gain})
        stacked.append(frame)
    write_table(pd.concat(metrics, ignore_index=True), tables / "ablation_metrics.csv")
    write_table(pd.DataFrame(gains), tables / "ablation_gain_vs_viirs.csv")
    write_table(pd.DataFrame(subgroups), tables / "ablation_subgroups.csv")
    write_table(pd.concat(stacked, ignore_index=True), tables / "ablation_predictions.csv")


if __name__ == "__main__":
    main()
