"""Validation externe entre enquêtes : DHS → EHCVM et EHCVM → DHS.

Un modèle entraîné sur une enquête est appliqué aux grappes de l'autre,
jamais vues et tirées indépendamment. C'est le test le plus exigeant de
généralisation. Les deux indices n'ayant pas la même échelle (hv271 DHS
contre ACP EHCVM), cible et prédiction sont centrées-réduites dans chaque
enquête ; on rapporte surtout la corrélation de rang (Spearman), invariante
à l'échelle.

Sortie : outputs/comparison/cross_survey_metrics.csv
"""
from pathlib import Path
import argparse
import sys

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd

from src.evaluation.common.compare_feature_sets import load_design
from src.utils.config import alpha_grid, load_config
from src.utils.feature_sets import uses_residual
from src.utils.io import write_table
from src.utils.metrics import regression_metrics
from src.utils.modeling import fit_final_ridge
from src.utils.paths import COMPARISON_OUTPUTS_DIR
from src.utils.spatial import merge_colocated, regular_blocks

OUTPUT_FILE = COMPARISON_OUTPUTS_DIR / "cross_survey_metrics.csv"


def zscore(values):
    return (values - values.mean()) / values.std()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sets", nargs="+", default=["viirs", "viirs5", "cnn_imagenet_tiles", "cnn_imagenet_tiles+viirs", "cnn_imagenet_tiles+viirs5"])
    args = parser.parse_args(argv)
    config = load_config()
    data = {survey: load_design(survey, args.sets) for survey in ("dhs", "ehcvm")}
    rows = []
    for train, test in [("dhs", "ehcvm"), ("ehcvm", "dhs")]:
        base_train, X_train = data[train]
        base_test, X_test = data[test]
        groups = merge_colocated(
            regular_blocks(base_train["latitude"], base_train["longitude"], config["validation"]["block_size_km"]),
            base_train["latitude"], base_train["longitude"])
        for spec in set(X_train) & set(X_test):
            model, alpha = fit_final_ridge(X_train[spec], zscore(base_train["wealth_mean"]), groups,
                                           alpha_grid(), config["validation"]["inner_folds"],
                                   residual=uses_residual(spec))
            prediction = model.predict(X_test[spec])
            rows.append({"train": train, "test": test, "feature_set": spec, "alpha": alpha,
                         **regression_metrics(zscore(base_test["wealth_mean"]), zscore(pd.Series(prediction)))})
            print(f"{train} → {test} | {spec:26s} Spearman = {rows[-1]['Spearman']:.3f}")
    write_table(pd.DataFrame(rows), OUTPUT_FILE)


if __name__ == "__main__":
    main()
