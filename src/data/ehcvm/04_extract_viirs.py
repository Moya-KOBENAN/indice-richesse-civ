"""EHCVM 04 — Lumières nocturnes VIIRS autour des grappes (Earth Engine).

Entrée  : data/interim/ehcvm/ehcvm_clusters_wealth_gps.csv (étape 03)
Sortie  : data/processed/ehcvm/ehcvm_clusters_wealth_gps_viirs.csv
          (table analytique EHCVM : une ligne par grappe)

Mêmes produits, tampons et statistiques que pour la DHS
(src/data/dhs/00_extract_viirs_dhs_gee.js) : 2 km urbain / 5 km rural,
5 km pour tous, 2 km urbain / 10 km rural.
Nécessite une authentification Earth Engine (`earthengine authenticate`).
"""
from pathlib import Path
import argparse
import sys

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd

from src.utils import earth_engine
from src.utils.config import load_config
from src.utils.io import check_full_merge, read_table, write_table
from src.utils.paths import EHCVM_INTERIM_DIR, ensure_layout, survey_clusters_file

INPUT_FILE = EHCVM_INTERIM_DIR / "ehcvm_clusters_wealth_gps.csv"
OUTPUT_FILE = survey_clusters_file("ehcvm")
BATCH_SIZE = 200  # limite la taille des requêtes getInfo


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Contrôler l'entrée sans appeler Earth Engine")
    args = parser.parse_args(argv)
    ensure_layout("ehcvm")
    config = load_config()
    clusters = read_table(INPUT_FILE, required=["cluster", "latitude", "longitude", "urban_rural"],
                          unique="cluster")
    if args.dry_run:
        print(f"{len(clusters)} grappes prêtes ; aucune requête Earth Engine envoyée.")
        return
    ee = earth_engine.initialize(config["earth_engine"]["project_id"])
    rows = []
    for start in range(0, len(clusters), BATCH_SIZE):
        batch = clusters.iloc[start:start + BATCH_SIZE]
        points = ee.FeatureCollection([
            ee.Feature(ee.Geometry.Point([row.longitude, row.latitude]),
                       {"cluster": int(row.cluster), "urban_rural": row.urban_rural})
            for row in batch.itertuples()])
        stats = earth_engine.viirs_buffer_stats(ee, points, config["ehcvm"]["viirs"], config["viirs"])
        rows += [feature["properties"] for feature in stats.getInfo()["features"]]
        print(f"VIIRS : {min(start + BATCH_SIZE, len(clusters))}/{len(clusters)} grappes")
    viirs = pd.DataFrame(rows).drop(columns="urban_rural")
    merged = check_full_merge(
        clusters.merge(viirs, on="cluster", how="outer", validate="one_to_one", indicator=True),
        label="grappes × VIIRS")
    print("VIIRS manquant :", int(merged.filter(like="_mean").isna().sum().sum()))
    write_table(merged, OUTPUT_FILE)


if __name__ == "__main__":
    main()
