"""DHS 02 — Coordonnées des grappes et fusion avec l'indice de richesse.

Entrées : data/interim/dhs/dhs_clusters_wealth.csv (étape 01)
          CIGE81FL.shp (GPS DHS, chemin dans config.yaml)
Sortie  : data/interim/dhs/DHS_CI_2021_clusters_richesse_GPS.csv

Les coordonnées DHS sont déplacées volontairement (jusqu'à 2 km en urbain,
5 km en rural, 10 km pour 1 % des grappes rurales) : c'est ce qui justifie
des zones d'image et des tampons VIIRS plus larges en milieu rural.
Les grappes sans GPS (LATNUM = 0) sont exclues et comptées.

Contrôle de reproductibilité : la table produite est comparée au fichier
historique data/raw/DHS_CI_2021_clusters_richesse_GPS.csv utilisé jusqu'ici.
"""
from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd

from src.utils.config import load_config, resolve_path
from src.utils.io import check_full_merge, read_table, write_table
from src.utils.paths import DHS_INTERIM_DIR, RAW_DIR, ensure_layout

WEALTH_FILE = DHS_INTERIM_DIR / "dhs_clusters_wealth.csv"
OUTPUT_FILE = DHS_INTERIM_DIR / "DHS_CI_2021_clusters_richesse_GPS.csv"
HISTORICAL_FILE = RAW_DIR / "DHS_CI_2021_clusters_richesse_GPS.csv"

GPS_COLUMNS = {"DHSCLUST": "cluster", "LATNUM": "latitude", "LONGNUM": "longitude",
               "URBAN_RURA": "urban_rural", "ADM1NAME": "adm1_name",
               "DHSREGNA": "dhs_region", "ALT_GPS": "altitude_gps", "ALT_DEM": "altitude_dem"}


def load_gps(path):
    import geopandas as gpd

    gps = pd.DataFrame(gpd.read_file(path).drop(columns="geometry"))
    gps = gps[list(GPS_COLUMNS)].rename(columns=GPS_COLUMNS)
    gps["cluster"] = gps["cluster"].astype(int)
    missing = (gps["latitude"] == 0) & (gps["longitude"] == 0)
    print(f"Grappes GPS : {len(gps)} | sans coordonnées exclues : {int(missing.sum())}")
    return gps.loc[~missing]


def compare_with_historical(table):
    """Signaler tout écart avec la table utilisée pour les résultats existants."""
    if not HISTORICAL_FILE.exists():
        print("Pas de fichier historique à comparer.")
        return
    old = pd.read_csv(HISTORICAL_FILE).set_index("cluster").sort_index()
    new = table.set_index("cluster").sort_index()
    common = [c for c in old.columns if c in new.columns and pd.api.types.is_numeric_dtype(old[c])]
    if not old.index.equals(new.index):
        print("ATTENTION : ensembles de grappes différents du fichier historique.")
        return
    gaps = (old[common] - new[common]).abs().max()
    print("Écart maximal avec le fichier historique :")
    print(gaps.to_string())


def main():
    ensure_layout("dhs")
    config = load_config()["dhs"]
    wealth = read_table(WEALTH_FILE, required=["cluster", "wealth_mean"], unique="cluster")
    gps = load_gps(resolve_path(config["raw_dir"]) / config["gps_file"])
    merged = wealth.merge(gps, on="cluster", how="outer", validate="one_to_one", indicator=True)
    no_gps = merged["_merge"].eq("left_only")
    if no_gps.any():
        print(f"Grappes sans GPS exclues de l'analyse spatiale : {merged.loc[no_gps, 'cluster'].tolist()}")
    merged = check_full_merge(merged.loc[~no_gps], label="richesse × GPS")
    columns = ["cluster", "n_households", "wealth_mean", "wealth_quintile_mean",
               "wealth_mean_weighted", "latitude", "longitude", "urban_rural",
               "adm1_name", "dhs_region", "altitude_gps", "altitude_dem"]
    table = merged[columns].sort_values("cluster")
    compare_with_historical(table)
    write_table(table, OUTPUT_FILE)


if __name__ == "__main__":
    main()
