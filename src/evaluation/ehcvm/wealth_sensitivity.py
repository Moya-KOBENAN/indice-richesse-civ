"""Sensibilité descriptive de l’ACP ; aucune métrique prédictive implicite."""
from pathlib import Path
import argparse
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
from src.utils.paths import EHCVM_INTERIM_DIR, EHCVM_TABLES_DIR
from src.utils.statistics import weighted_correlation
from src.utils.wealth import positive_weights, weighted_pca_first_component


def sensitivity(frame, params):
    weights = positive_weights(frame.hhweight, len(frame))
    if not np.isfinite(frame[['hhsize', 's11q02']].to_numpy()).all() or (frame[['hhsize', 's11q02']] <= 0).any().any():
        raise ValueError("La sensibilité exige tailles et nombres de pièces observés et positifs.")
    assets = select_assets(frame, weights, params['min_asset_prevalence'], params['max_asset_prevalence'])
    if assets.isna().any().any():
        raise ValueError("Actifs incomplets : appliquer une politique explicite avant la sensibilité.")
    rooms = np.log(frame.s11q02 / frame.hhsize).rename('log_pieces_par_personne')
    housing = housing_dummies(frame, params['housing_variables'], weights, params['min_category_share'])
    designs = {'reference': pd.concat([housing, assets, rooms], axis=1),
               'assets_only': assets, 'housing_only': pd.concat([housing, rooms], axis=1)}
    for threshold in (.005, .02):
        design = housing_dummies(frame, params['housing_variables'], weights, threshold)
        designs[f'housing_threshold_{threshold}'] = pd.concat([design, assets, rooms], axis=1)
    reference, rows = None, []
    for name, design in designs.items():
        score, _, explained, kept = weighted_pca_first_component(design.to_numpy(float), weights)
        if weighted_correlation(score, assets.sum(axis=1), weights) < 0:
            score = -score
        if reference is None:
            reference = score
        rows.append({'scenario': name, 'spearman_reference_unweighted': spearmanr(reference, score).statistic,
                     'first_axis_variance_share': explained[0], 'n_variables': int(kept.sum()),
                     'n_households': len(frame)})
    return pd.DataFrame(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    frame = read_table(EHCVM_INTERIM_DIR / 'ehcvm_households.csv',
                       required=['hhweight', 'hhsize', 's11q02'])
    if args.dry_run:
        return
    result = sensitivity(frame, load_config()['ehcvm']['wealth_index'])
    write_table(result, EHCVM_TABLES_DIR / 'wealth_index_sensitivity.csv')


if __name__ == "__main__":
    main()
