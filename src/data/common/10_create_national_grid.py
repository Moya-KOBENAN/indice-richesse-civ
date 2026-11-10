"""Étape historique isolée : exécution explicite uniquement via main()."""


def main():
    # Résolution des chemins indépendante du répertoire de lancement.
    from pathlib import Path
    import sys

    _PROJECT_ROOT = Path(__file__).resolve().parents[3]
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))

    from src.utils.paths import (
        COMMON_PROCESSED_DIR,
        PROJECT_ROOT,
        RAW_DIR,
        ensure_layout,
    )

    ensure_layout('common')


    import geopandas as gpd
    import numpy as np
    from shapely.geometry import Point


    # ============================================================
    # CONFIGURATION
    # ============================================================

    BOUNDARY_FILE = RAW_DIR / 'cote_ivoire_boundary.geojson'
    OUTPUT_FILE = COMMON_PROCESSED_DIR / 'cote_ivoire_prediction_grid.geojson'

    # Espacement initial entre les points de prédiction
    GRID_SPACING_KM = 10

    # Projection métrique adaptée à la Côte d'Ivoire
    METRIC_CRS = "EPSG:32630"


    # ============================================================
    # CHARGEMENT DE LA FRONTIERE
    # ============================================================

    print("=" * 72)
    print("ETAPE 16.1 - CREATION DE LA GRILLE NATIONALE")
    print("=" * 72)

    if not BOUNDARY_FILE.exists():
        raise FileNotFoundError(
            f"Frontière introuvable : {BOUNDARY_FILE}"
        )

    country = gpd.read_file(BOUNDARY_FILE)

    if country.empty:
        raise ValueError("Le fichier de frontière est vide.")

    country = country.to_crs(METRIC_CRS)

    print(f"Géométries frontière : {len(country)}")


    # ============================================================
    # CREATION DE LA GRILLE
    # ============================================================

    boundary = country.geometry.union_all()

    minx, miny, maxx, maxy = boundary.bounds

    spacing_m = GRID_SPACING_KM * 1000

    x_coords = np.arange(
        minx + spacing_m / 2,
        maxx,
        spacing_m
    )

    y_coords = np.arange(
        miny + spacing_m / 2,
        maxy,
        spacing_m
    )

    points = []

    for x in x_coords:
        for y in y_coords:
            point = Point(x, y)

            if boundary.covers(point):
                points.append(point)

    grid = gpd.GeoDataFrame(
        {
            "grid_id": np.arange(1, len(points) + 1),
            "spacing_km": GRID_SPACING_KM,
        },
        geometry=points,
        crs=METRIC_CRS,
    )

    print(f"Points conservés dans le territoire : {len(grid)}")


    # ============================================================
    # COORDONNEES GPS
    # ============================================================

    grid_wgs84 = grid.to_crs("EPSG:4326")

    grid_wgs84["longitude"] = grid_wgs84.geometry.x
    grid_wgs84["latitude"] = grid_wgs84.geometry.y

    # Contrôles
    if grid_wgs84["grid_id"].duplicated().any():
        raise ValueError("Des grid_id dupliqués ont été détectés.")

    if grid_wgs84[["longitude", "latitude"]].isna().any().any():
        raise ValueError("Coordonnées manquantes détectées.")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    grid_wgs84.to_file(
        OUTPUT_FILE,
        driver="GeoJSON"
    )


    # ============================================================
    # RESUME
    # ============================================================

    print()
    print("Contrôles :")
    print(f"  Grid ID uniques : {grid_wgs84['grid_id'].nunique()}")
    print(
        f"  Longitude : "
        f"{grid_wgs84['longitude'].min():.4f} à "
        f"{grid_wgs84['longitude'].max():.4f}"
    )
    print(
        f"  Latitude  : "
        f"{grid_wgs84['latitude'].min():.4f} à "
        f"{grid_wgs84['latitude'].max():.4f}"
    )

    print()
    print(f"Fichier créé : {OUTPUT_FILE}")

    print()
    print("=" * 72)
    print("ETAPE 16.1 TERMINEE")
    print("=" * 72)

if __name__ == "__main__":
    main()
