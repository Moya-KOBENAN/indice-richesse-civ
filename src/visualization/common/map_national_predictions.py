"""Carte expérimentale de la richesse prédite sur la grille nationale.

Entrée : outputs/common/tables/national_wealth_predictions.csv
Sortie : outputs/common/maps/carte_richesse_predite.png

Chaque point de grille représente une zone de 5 km × 5 km (espacement de
10 km). Les points hors de l'intervalle d'apprentissage sont entourés : la
prédiction y est une extrapolation.
"""
from pathlib import Path
import sys

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.utils.io import read_table
from src.utils.paths import BOUNDARY_FILE, COMMON_OUTPUTS_DIR, COMMON_TABLES_DIR, ensure_layout

INPUT_FILE = COMMON_TABLES_DIR / "national_wealth_predictions.csv"
OUTPUT_FILE = COMMON_OUTPUTS_DIR / "maps" / "carte_richesse_predite.png"


def main():
    import geopandas as gpd

    ensure_layout("common")
    grid = read_table(INPUT_FILE, required=["longitude", "latitude", "wealth_predicted", "outside_training_range"])
    boundary = gpd.read_file(BOUNDARY_FILE)
    fig, ax = plt.subplots(figsize=(9, 9))
    boundary.boundary.plot(ax=ax, color="#444444", linewidth=0.8)
    points = ax.scatter(grid["longitude"], grid["latitude"], c=grid["wealth_predicted"],
                        cmap="viridis", s=14, marker="s")
    outside = grid[grid["outside_training_range"]]
    ax.scatter(outside["longitude"], outside["latitude"], facecolors="none",
               edgecolors="#d62728", s=30, linewidths=0.6, label="Hors domaine d'apprentissage")
    fig.colorbar(points, ax=ax, shrink=0.7, label="Indice de richesse prédit")
    ax.set(title=f"Richesse prédite — {grid['model'].iloc[0]}\n(sortie de modèle, non officielle)",
           xlabel="Longitude", ylabel="Latitude")
    ax.set_aspect("equal")
    ax.legend(loc="lower left")
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_FILE, dpi=200, bbox_inches="tight")
    print(f"Carte : {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
