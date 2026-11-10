"""Commun 10d — VIIRS sur la grille nationale, définie comme pour les grappes.

Entrée  : data/processed/common/cote_ivoire_national_grid_viirs_diagnostic.csv
Sortie  : data/processed/common/cote_ivoire_national_grid_viirs_buffers.csv

La variable VIIRS de l'étape 10b (tampon de 2,5 km) sert à l'adaptation du
CNN, pas à la prédiction. Pour appliquer un modèle entraîné sur les grappes,
la grille doit porter exactement la même variable. Seul le tampon de 5 km
pour tous (`viirs_5all_*`, variante « viirs5 ») se définit sans connaître le
milieu urbain/rural : c'est la seule conservée ici.
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
from src.utils.paths import COMMON_PROCESSED_DIR, ensure_layout

INPUT_FILE = COMMON_PROCESSED_DIR / "cote_ivoire_national_grid_viirs_diagnostic.csv"
OUTPUT_FILE = COMMON_PROCESSED_DIR / "cote_ivoire_national_grid_viirs_buffers.csv"
BATCH_SIZE = 250


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    ensure_layout("common")
    config = load_config()
    grid = read_table(INPUT_FILE, required=["grid_id", "latitude", "longitude"], unique="grid_id")
    if args.dry_run:
        print(f"{len(grid)} points de grille prêts ; aucune requête Earth Engine envoyée.")
        return
    ee = earth_engine.initialize(config["earth_engine"]["project_id"])
    rows = []
    for start in range(0, len(grid), BATCH_SIZE):
        batch = grid.iloc[start:start + BATCH_SIZE]
        # Milieu inconnu sur la grille : seul le tampon 5 km pour tous est conservé.
        points = ee.FeatureCollection([
            ee.Feature(ee.Geometry.Point([row.longitude, row.latitude]),
                       {"grid_id": int(row.grid_id), "urban_rural": "R"})
            for row in batch.itertuples()])
        stats = earth_engine.viirs_buffer_stats(ee, points, config["ehcvm"]["viirs"], config["viirs"])
        rows += [feature["properties"] for feature in stats.getInfo()["features"]]
        print(f"VIIRS grille : {min(start + BATCH_SIZE, len(grid))}/{len(grid)}", flush=True)
    viirs = pd.DataFrame(rows)
    viirs = viirs[["grid_id"] + [c for c in viirs.columns if c.startswith("viirs_5all_")]]
    merged = check_full_merge(
        grid.merge(viirs, on="grid_id", how="outer", validate="one_to_one", indicator=True),
        label="grille × VIIRS")
    print("VIIRS manquant :", int(merged["viirs_5all_mean"].isna().sum()))
    write_table(merged, OUTPUT_FILE)


if __name__ == "__main__":
    main()
