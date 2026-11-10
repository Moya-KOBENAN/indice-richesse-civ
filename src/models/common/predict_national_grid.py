"""Prédiction de l'indice de richesse sur la grille nationale (10 km).

Modèle : Ridge entraînée sur toutes les grappes de l'enquête choisie, alpha
choisi par validation spatiale par blocs. Le jeu de variables
(config.yaml : prediction.feature_set) doit exister pour les grappes ET
pour la grille, extrait avec le même CNN et le même découpage.

Les lumières nocturnes de la grille doivent avoir la même définition que
celles des grappes : l'étape common/10d calcule les tampons des grappes ; seule
la variante « viirs5 » (5 km pour tous) est définie sans connaître le milieu.

Sorties : outputs/common/tables/national_wealth_predictions.csv
          outputs/common/models/ridge_national_<survey>_<jeu>.joblib

Interprétation : prédiction expérimentale d'un modèle, pas une statistique
officielle ; elle n'est pertinente que si le R² spatial de l'ablation l'est.
"""
from pathlib import Path
import argparse
import sys

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd

from src.features.design import load_design, feature_columns
from src.utils.config import alpha_grid, load_config
from src.utils.feature_sets import cnn_file, parse_feature_set, uses_residual
from src.utils.io import read_table, write_table
from src.utils.modeling import fit_final_ridge
from src.utils.paths import COMMON_MODELS_DIR, COMMON_PROCESSED_DIR, COMMON_TABLES_DIR, ensure_layout
from src.utils.spatial import merge_colocated, regular_blocks

GRID_FILE = COMMON_PROCESSED_DIR / "cote_ivoire_national_grid_viirs_buffers.csv"
OUTPUT_FILE = COMMON_TABLES_DIR / "national_wealth_predictions.csv"


def grid_design(spec):
    grid = read_table(GRID_FILE, required=["grid_id", "latitude", "longitude"], unique="grid_id")
    cnn_names, viirs_column = parse_feature_set(spec)
    parts = []
    for name in cnn_names:
        features = read_table(cnn_file("grid", name), required=["grid_id"], unique="grid_id")
        grid = grid.merge(features[["grid_id"]], on="grid_id", how="inner", validate="one_to_one")
        parts.append(feature_columns(features.set_index("grid_id"), cnn_file("grid", name)))
    if viirs_column and viirs_column not in grid.columns:
        raise ValueError(f"La grille n'a pas {viirs_column} : exécuter common/10d "
                         "ou choisir un jeu sans VIIRS dans prediction.feature_set.")
    X = np.hstack([p.loc[grid["grid_id"]].to_numpy() for p in parts]
                  + ([np.log1p(grid[[viirs_column]].clip(lower=0).to_numpy())] if viirs_column else []))
    return grid, X


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    config = load_config()
    survey, spec = config["prediction"]["survey"], config["prediction"]["feature_set"]
    base, matrices = load_design(survey, [spec])
    if spec not in matrices:
        raise SystemExit(f"Caractéristiques « {spec} » absentes pour {survey} : exécuter d'abord l'extraction (06t/13t).")
    grid, X_grid = grid_design(spec)
    print(f"Apprentissage : {len(base)} grappes {survey} | grille : {len(grid)} points | jeu : {spec}")
    if args.dry_run:
        return
    ensure_layout("common")
    import joblib

    groups = merge_colocated(regular_blocks(base["latitude"], base["longitude"],
                                            config["validation"]["block_size_km"]),
                             base["latitude"], base["longitude"])
    model, alpha = fit_final_ridge(matrices[spec], base["wealth_mean"], groups, alpha_grid(),
                                   config["validation"]["inner_folds"],
                                   residual=uses_residual(spec))
    grid["wealth_predicted"] = model.predict(X_grid)
    # Diagnostic de l’étendue des scores uniquement ; ce n’est pas un test hors distribution.
    low, high = base["wealth_mean"].quantile([0.01, 0.99])
    grid["outside_training_range"] = ~grid["wealth_predicted"].between(low, high)
    grid["model"] = f"ridge_{survey}_{spec}"
    grid["alpha"] = alpha
    write_table(grid[["grid_id", "latitude", "longitude", "wealth_predicted",
                      "outside_training_range", "model", "alpha"]], OUTPUT_FILE)
    joblib.dump(model, COMMON_MODELS_DIR / f"ridge_national_{survey}_{spec.replace('+', '_')}.joblib")


if __name__ == "__main__":
    main()
