"""EHCVM 03 — Agrégation de l'indice par grappe (unité d'analyse).

Entrée  : data/interim/ehcvm/ehcvm_households_wealth.csv (étape 02)
Sortie  : data/interim/ehcvm/ehcvm_clusters_wealth_gps.csv

Colonnes alignées sur la table DHS (cluster, wealth_mean, latitude,
longitude, urban_rural...) pour que les étapes communes (caractéristiques,
évaluation, carte) traitent les deux enquêtes de la même façon.
La cible principale est la moyenne pondérée par hhweight ; contrairement à
la DHS, les poids varient à l'intérieur d'une grappe EHCVM.
"""
from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd

from src.utils.config import load_config
from src.utils.io import read_table, write_table
from src.utils.paths import EHCVM_INTERIM_DIR, ensure_layout

INPUT_FILE = EHCVM_INTERIM_DIR / "ehcvm_households_wealth.csv"
OUTPUT_FILE = EHCVM_INTERIM_DIR / "ehcvm_clusters_wealth_gps.csv"
CONSTANT_WITHIN_CLUSTER = ["vague", "milieu", "region", "latitude", "longitude"]


def aggregate(households, urban_code):
    varying = households.groupby("grappe")[CONSTANT_WITHIN_CLUSTER].nunique().gt(1).any()
    if varying.any():
        raise ValueError(f"Variables non constantes dans une grappe : {varying[varying].index.tolist()}")
    grouped = households.groupby("grappe")
    clusters = grouped[CONSTANT_WITHIN_CLUSTER].first()
    clusters["n_households"] = grouped.size()
    clusters["wealth_mean"] = grouped.apply(
        lambda g: np.average(g["wealth_score"], weights=g["hhweight"]), include_groups=False)
    clusters["wealth_mean_unweighted"] = grouped["wealth_score"].mean()
    clusters["wealth_quintile_mean"] = grouped["wealth_quintile"].mean()
    clusters["log_pcexp_mean"] = grouped.apply(
        lambda g: np.average(np.log(g["pcexp"]), weights=g["hhweight"]), include_groups=False)
    clusters["urban_rural"] = np.where(clusters["milieu"] == urban_code, "U", "R")
    clusters = clusters.reset_index().rename(columns={"grappe": "cluster"})
    position = clusters["latitude"].round(6).astype(str) + "_" + clusters["longitude"].round(6).astype(str)
    clusters["shares_position"] = position.duplicated(keep=False)
    return clusters


def main():
    ensure_layout("ehcvm")
    config = load_config()["ehcvm"]
    households = read_table(INPUT_FILE, required=["grappe", "wealth_score", "hhweight"] + CONSTANT_WITHIN_CLUSTER)
    clusters = aggregate(households, config["urban_code"])
    if len(clusters) != config["expected_clusters"]:
        print(f"ATTENTION : {len(clusters)} grappes au lieu de {config['expected_clusters']}.")
    print(clusters[["n_households", "wealth_mean"]].describe().to_string())
    print(f"Grappes partageant une position : {int(clusters['shares_position'].sum())}")
    write_table(clusters, OUTPUT_FILE)


if __name__ == "__main__":
    main()
