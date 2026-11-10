"""EHCVM 05 — Images Sentinel-2 de 5 km × 5 km centrées sur chaque grappe.

Entrée  : data/processed/ehcvm/ehcvm_clusters_wealth_gps_viirs.csv (étape 04)
Sorties : data/raw/ehcvm/satellite/cluster_<id>_sentinel2.tif
          outputs/ehcvm/tables/satellite_download_log.csv

Même composite que la DHS (médiane, masque SCL, bandes B4-B3-B2, 10 m),
sur la période des deux vagues EHCVM (config.yaml : ehcvm.satellite).
Reprise automatique : une image déjà présente et valide n'est pas
retéléchargée ; le journal est écrit après chaque grappe.
"""
from pathlib import Path
import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd

from src.utils import earth_engine
from src.utils.config import load_config
from src.utils.io import read_table
from src.utils.paths import EHCVM_SATELLITE_DIR, EHCVM_TABLES_DIR, ensure_layout, survey_clusters_file

INPUT_FILE = survey_clusters_file("ehcvm")
LOG_FILE = EHCVM_TABLES_DIR / "satellite_download_log.csv"


def image_path(cluster):
    return EHCVM_SATELLITE_DIR / f"cluster_{int(cluster):04d}_sentinel2.tif"


def process(ee, row, sat):
    output = image_path(row.cluster)
    record = {"cluster": int(row.cluster)}
    if output.exists():
        quality = earth_engine.patch_quality(output, min_nonzero_ratio=sat["minimum_nonzero_ratio"])
        if quality["valid"]:
            return {**record, "status": "DEJA_OK", **quality}
        output.unlink()
    region = ee.Geometry.Point([row.longitude, row.latitude]).buffer(sat["patch_radius_m"]).bounds()
    composite, n_images = earth_engine.sentinel2_composite(ee, region, sat)
    if composite is None:
        return {**record, "status": "ECHEC_AUCUNE_IMAGE", "n_images": 0}
    earth_engine.download_patch(ee, composite, region, sat["scale_m"], output)
    quality = earth_engine.patch_quality(output, min_nonzero_ratio=sat["minimum_nonzero_ratio"])
    return {**record, "status": "OK" if quality["valid"] else "ECHEC_QUALITE",
            "n_images": n_images, **quality}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, help="Traiter seulement les N premières grappes (essai)")
    parser.add_argument("--workers", type=int, default=6,
                        help="Téléchargements simultanés (Earth Engine accepte des requêtes parallèles)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    ensure_layout("ehcvm")
    config = load_config()
    sat = {**config["satellite"], **config["ehcvm"]["satellite"]}
    clusters = read_table(INPUT_FILE, required=["cluster", "latitude", "longitude"], unique="cluster")
    if args.limit:
        clusters = clusters.head(args.limit)
    if args.dry_run:
        print(f"{len(clusters)} grappes ; période {sat['start_date']} → {sat['end_date']} ; "
              f"sortie {EHCVM_SATELLITE_DIR}")
        return
    ee = earth_engine.initialize(config["earth_engine"]["project_id"])
    log = []

    def safe_process(row):
        try:
            return process(ee, row, sat)
        except Exception as error:  # une grappe en échec ne bloque pas les autres
            return {"cluster": int(row.cluster), "status": "ERREUR", "error": str(error)}

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(safe_process, row) for row in clusters.itertuples()]
        for position, future in enumerate(as_completed(futures), start=1):
            log.append(future.result())
            pd.DataFrame(log).to_csv(LOG_FILE, index=False)
            print(f"{position}/{len(clusters)} grappe {log[-1]['cluster']} : {log[-1]['status']}", flush=True)
    summary = pd.DataFrame(log)["status"].value_counts()
    print(summary.to_string())


if __name__ == "__main__":
    main()
