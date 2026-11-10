"""Réplication des deux analyses phares de Jean et al. (2016, Science).

1. Test placebo (Jean, fig. 4C-D) : les images sont réattribuées au hasard
   aux grappes, puis le modèle est réentraîné. Si les caractéristiques CNN
   portaient une information réelle, le R² obtenu avec les vraies images doit
   dépasser nettement toute la distribution placebo. p = (k + 1) / (n + 1),
   où k est le nombre d'essais placebo au moins aussi bons que le vrai.

2. Performance chez les plus pauvres (Jean, fig. 4A-B) : l'échantillon est
   restreint aux grappes sous le quantile q de richesse (q = 20 %… 100 %) et
   chaque modèle est réentraîné sur ce sous-échantillon. C'est là que les
   lumières nocturnes varient peu et où l'imagerie de jour est censée aider.

Protocole : Ridge imbriquée, comme dans l'ablation ; validation aléatoire
(celle de Jean) et par blocs de 100 km (critère principal du projet).

Sorties (outputs/<enquête>/tables/) : jean_placebo.csv, jean_quantiles.csv
"""
from pathlib import Path
import argparse
import sys

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd

from src.evaluation.common.compare_feature_sets import load_design
from src.utils.config import alpha_grid, load_config
from src.utils.feature_sets import uses_residual
from src.utils.io import write_table
from src.utils.metrics import regression_metrics
from src.utils.modeling import nested_ridge_predictions
from src.utils.paths import SURVEYS, survey_dirs
from src.utils.spatial import outer_splits

PLACEBO_SET = "cnn_imagenet_tiles"
QUANTILE_SETS = ["viirs", "cnn_imagenet_tiles", "cnn_imagenet_tiles+viirs"]
QUANTILES = [0.2, 0.4, 0.6, 0.8, 1.0]


def r2(base, X, spec, scheme, config):
    splits, groups = outer_splits(scheme, base, config)
    y = base["wealth_mean"].to_numpy(float)
    pred, _, _ = nested_ridge_predictions(X, y, splits, groups, alpha_grid(),
                                          config["validation"]["inner_folds"], uses_residual(spec))
    return regression_metrics(y, pred)["R2"]


def placebo(base, X, config, n_trials, scheme):
    """R² réel et distribution des R² avec images permutées entre grappes."""
    rng = np.random.default_rng(config["project"]["random_state"])
    true_r2 = r2(base, X, PLACEBO_SET, scheme, config)
    trials = []
    for trial in range(1, n_trials + 1):
        trials.append(r2(base, X[rng.permutation(len(X))], PLACEBO_SET, scheme, config))
        if trial % 10 == 0:
            print(f"  placebo {scheme} : {trial}/{n_trials} (max placebo = {max(trials):.3f})", flush=True)
    trials = np.array(trials)
    return {"scheme": scheme, "feature_set": PLACEBO_SET, "R2_vraies_images": true_r2,
            "R2_placebo_moyen": trials.mean(), "R2_placebo_max": trials.max(),
            "R2_placebo_p95": np.percentile(trials, 95), "n_essais": n_trials,
            "p_valeur": (np.sum(trials >= true_r2) + 1) / (n_trials + 1)}


def by_quantile(base, matrices, config, schemes):
    """R² de chaque modèle réentraîné sur les grappes sous chaque quantile."""
    rows = []
    thresholds = base["wealth_mean"].quantile(QUANTILES)
    for q, threshold in zip(QUANTILES, thresholds):
        keep = (base["wealth_mean"] <= threshold).to_numpy()
        subset = base.loc[keep].reset_index(drop=True)
        for scheme in schemes:
            for spec in QUANTILE_SETS:
                value = r2(subset, matrices[spec][keep], spec, scheme, config)
                rows.append({"quantile": q, "n_grappes": int(keep.sum()), "scheme": scheme,
                             "feature_set": spec, "R2": value})
                print(f"  q ≤ {int(q * 100):3d} % | {scheme:6s} | {spec:26s} R² = {value:.3f}", flush=True)
    return pd.DataFrame(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--survey", choices=SURVEYS, default="dhs")
    parser.add_argument("--placebo-trials", type=int, default=None,
                        help="Nombre d'essais placebo (défaut : config validation.placebo_trials)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    config = load_config()
    n_trials = args.placebo_trials or config["validation"]["placebo_trials"]
    base, matrices = load_design(args.survey, sorted(set(QUANTILE_SETS + [PLACEBO_SET])))
    missing = [s for s in set(QUANTILE_SETS + [PLACEBO_SET]) if s not in matrices]
    if missing:
        raise SystemExit(f"Caractéristiques absentes : {missing}. Lancer d'abord 06t.")
    print(f"{args.survey} : {len(base)} grappes ; {n_trials} essais placebo par schéma")
    if args.dry_run:
        return
    tables = survey_dirs(args.survey)["tables"]
    schemes = ["random", "blocks"]
    write_table(by_quantile(base, matrices, config, schemes), tables / "jean_quantiles.csv")
    rows = [placebo(base, matrices[PLACEBO_SET], config, n_trials, scheme) for scheme in schemes]
    write_table(pd.DataFrame(rows), tables / "jean_placebo.csv")


if __name__ == "__main__":
    main()
