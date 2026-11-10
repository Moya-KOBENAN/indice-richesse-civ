"""DHS 01 — Indice de richesse par grappe à partir du fichier ménages (HR).

Entrée  : CIHR81FL.DTA (DHS Côte d'Ivoire 2021), chemin dans config.yaml.
Sortie  : data/interim/dhs/dhs_clusters_wealth.csv, une ligne par grappe.

Cible : hv271, score factoriel de richesse calculé par le programme DHS
(première composante d'une ACP sur les biens et le logement). On ne le
recalcule pas : c'est l'indice officiel, comparable à la littérature.
Moyenne simple et moyenne pondérée (hv005) sont conservées ; en DHS le poids
est constant à l'intérieur d'une grappe, les deux coïncident (contrôlé ici).
"""
from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd

from src.utils.config import load_config, resolve_path
from src.utils.io import write_table
from src.utils.paths import DHS_INTERIM_DIR, ensure_layout

OUTPUT_FILE = DHS_INTERIM_DIR / "dhs_clusters_wealth.csv"
COLUMNS = ["hv001", "hv005", "hv270", "hv271"]


def aggregate_clusters(households, divisor):
    """Moyennes par grappe du score de richesse et du quintile."""
    households = households.assign(score=households["hv271"] / divisor,
                                   weight=households["hv005"] / 1e6)
    grouped = households.groupby("hv001")
    clusters = pd.DataFrame({
        "n_households": grouped.size(),
        "wealth_mean": grouped["score"].mean(),
        "wealth_quintile_mean": grouped["hv270"].mean(),
        "wealth_mean_weighted": grouped.apply(
            lambda g: np.average(g["score"], weights=g["weight"]), include_groups=False),
        "wealth_sd": grouped["score"].std(),
    }).reset_index().rename(columns={"hv001": "cluster"})
    return clusters


def main():
    ensure_layout("dhs")
    config = load_config()["dhs"]
    source = resolve_path(config["raw_dir"]) / config["household_file"]
    households = pd.read_stata(source, columns=COLUMNS, convert_categoricals=False)
    if households[COLUMNS].isna().any().any():
        raise ValueError("Valeurs manquantes dans les variables DHS clés.")
    clusters = aggregate_clusters(households, config["wealth_score_divisor"])
    if len(clusters) != config["expected_clusters"]:
        raise ValueError(f"{len(clusters)} grappes au lieu de {config['expected_clusters']}.")
    gap = (clusters["wealth_mean"] - clusters["wealth_mean_weighted"]).abs().max()
    print(f"Ménages : {len(households)} | grappes : {len(clusters)} | écart max simple/pondéré : {gap:.2e}")
    write_table(clusters, OUTPUT_FILE)


if __name__ == "__main__":
    main()
