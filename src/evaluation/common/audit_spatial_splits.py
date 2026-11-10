"""Distances apprentissage–test, sans publier les positions des grappes."""
from pathlib import Path
import argparse
import sys

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from src.utils.config import load_config
from src.utils.io import read_table, write_table
from src.utils.paths import survey_clusters_file, survey_dirs
from src.utils.spatial import outer_splits, to_local_km


def distance_diagnostics(frame, splits, image_side_km):
    xy = np.column_stack(to_local_km(frame.latitude, frame.longitude))
    rows = []
    for fold, (train, test) in enumerate(splits, 1):
        nearest = NearestNeighbors(n_neighbors=1).fit(xy[train]).kneighbors(xy[test])[0][:, 0]
        rows.append({'fold': fold, 'n_train': len(train), 'n_test': len(test),
                     'nearest_train_min_km': nearest.min(), 'nearest_train_median_km': np.median(nearest),
                     'share_below_image_side': (nearest < image_side_km).mean()})
    return pd.DataFrame(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--survey', choices=['dhs', 'ehcvm'], required=True)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    frame = read_table(survey_clusters_file(args.survey), required=['cluster', 'latitude', 'longitude'], unique='cluster')
    if args.dry_run:
        return
    config = load_config()
    rows = []
    for scheme in ('random', 'blocks', 'kmeans'):
        splits, _ = outer_splits(scheme, frame, config)
        rows.append(distance_diagnostics(frame, splits, config['satellite']['patch_size_km']).assign(scheme=scheme))
    write_table(pd.concat(rows, ignore_index=True), survey_dirs(args.survey)['tables'] / 'spatial_distance_diagnostics.csv')


if __name__ == "__main__":
    main()
