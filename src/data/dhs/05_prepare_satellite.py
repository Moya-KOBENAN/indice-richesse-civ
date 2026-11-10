"""Étape historique isolée : exécution explicite uniquement via main()."""


def main():
    # Résolution des chemins indépendante du répertoire de lancement.
    from pathlib import Path
    import sys

    _PROJECT_ROOT = Path(__file__).resolve().parents[3]
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))

    from src.utils.paths import (
        DHS_PROCESSED_DIR,
        DHS_SATELLITE_DIR,
        DHS_TABLES_DIR,
        PROJECT_ROOT,
        ensure_layout,
    )

    ensure_layout('dhs')

    import io
    import time
    import zipfile

    import ee
    import numpy as np
    import pandas as pd
    import rasterio
    import requests


    # ============================================================
    # PROJET INDICE DE RICHESSE - COTE D'IVOIRE
    # ETAPE 5 : EXTRACTION SENTINEL-2
    # VERSION COMPLETE - 539 CLUSTERS DHS
    # ============================================================

    from src.utils.config import load_config

    PROJECT_ID = load_config()["earth_engine"]["project_id"]

    BASE_DIR = PROJECT_ROOT

    INPUT_FILE = (
        DHS_PROCESSED_DIR / 'DHS_CI_2021_richesse_GPS_VIIRS_merged.csv'
    )

    OUTPUT_DIR = DHS_SATELLITE_DIR

    LOG_FILE = (
        DHS_TABLES_DIR / 'satellite_download_log.csv'
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    PATCH_RADIUS_METERS = 2500
    SCALE_METERS = 10

    START_DATE = "2021-01-01"
    END_DATE = "2022-01-01"


    # ============================================================
    # 1. EARTH ENGINE
    # ============================================================

    ee.Initialize(project=PROJECT_ID)

    print("Connexion Earth Engine : OK")


    # ============================================================
    # 2. DONNEES DHS
    # ============================================================

    df = pd.read_csv(INPUT_FILE)

    required = [
        "cluster",
        "latitude",
        "longitude",
        "urban_rural"
    ]

    missing = [
        col for col in required
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Colonnes manquantes : {missing}"
        )

    print("Nombre total de clusters :", len(df))


    # ============================================================
    # 3. MASQUE SENTINEL-2
    # ============================================================

    def mask_s2(image):

        scl = image.select("SCL")

        mask = (
            scl.neq(0)
            .And(scl.neq(1))
            .And(scl.neq(3))
            .And(scl.neq(8))
            .And(scl.neq(9))
            .And(scl.neq(10))
            .And(scl.neq(11))
        )

        return (
            image
            .updateMask(mask)
            .select(["B4", "B3", "B2"])
            .divide(10000)
        )


    # ============================================================
    # 4. COMPOSITE LOCAL
    # ============================================================

    def create_local_composite(region):

        collection = (
            ee.ImageCollection(
                "COPERNICUS/S2_SR_HARMONIZED"
            )
            .filterDate(START_DATE, END_DATE)
            .filterBounds(region)
            .filter(
                ee.Filter.lt(
                    "CLOUDY_PIXEL_PERCENTAGE",
                    40
                )
            )
        )

        n_images = collection.size().getInfo()

        # Deuxième tentative si couverture insuffisante
        if n_images < 3:

            collection = (
                ee.ImageCollection(
                    "COPERNICUS/S2_SR_HARMONIZED"
                )
                .filterDate(START_DATE, END_DATE)
                .filterBounds(region)
                .filter(
                    ee.Filter.lt(
                        "CLOUDY_PIXEL_PERCENTAGE",
                        80
                    )
                )
            )

            n_images = collection.size().getInfo()

        if n_images == 0:
            return None, 0

        composite = (
            collection
            .map(mask_s2)
            .median()
        )

        return composite, n_images


    # ============================================================
    # 5. CONTROLE QUALITE
    # ============================================================

    def check_quality(file_path):

        try:

            with rasterio.open(file_path) as src:

                data = src.read(masked=True)

                width = src.width
                height = src.height
                bands = src.count

                if data.count() == 0:

                    return {
                        "valid": False,
                        "width": width,
                        "height": height,
                        "bands": bands,
                        "pct_non_zero": 0.0,
                        "mean_reflectance": 0.0
                    }

                values = data.compressed()

                if len(values) == 0:

                    return {
                        "valid": False,
                        "width": width,
                        "height": height,
                        "bands": bands,
                        "pct_non_zero": 0.0,
                        "mean_reflectance": 0.0
                    }

                pct_non_zero = (
                    100
                    * np.count_nonzero(values)
                    / len(values)
                )

                mean_reflectance = float(
                    np.mean(values)
                )

                valid = (
                    bands == 3
                    and width >= 450
                    and height >= 450
                    and pct_non_zero >= 90
                    and mean_reflectance > 0
                )

                return {
                    "valid": valid,
                    "width": width,
                    "height": height,
                    "bands": bands,
                    "pct_non_zero": pct_non_zero,
                    "mean_reflectance": mean_reflectance
                }

        except Exception:

            return {
                "valid": False,
                "width": 0,
                "height": 0,
                "bands": 0,
                "pct_non_zero": 0.0,
                "mean_reflectance": 0.0
            }


    # ============================================================
    # 6. SAUVEGARDE DU JOURNAL
    # ============================================================

    def save_log(results):

        pd.DataFrame(results).to_csv(
            LOG_FILE,
            index=False
        )


    # ============================================================
    # 7. TELECHARGEMENT
    # ============================================================

    def process_cluster(row):

        cluster = int(row["cluster"])
        lat = float(row["latitude"])
        lon = float(row["longitude"])
        milieu = row["urban_rural"]

        output_file = (
            OUTPUT_DIR
            / f"cluster_{cluster:03d}_sentinel2_2021.tif"
        )

        print("\n------------------------------------")
        print(f"Cluster {cluster}")
        print(f"Coordonnées : {lat:.4f}, {lon:.4f}")
        print(f"Milieu : {milieu}")

        # --------------------------------------------------------
        # Vérifier si le fichier existe déjà
        # --------------------------------------------------------

        if output_file.exists():

            quality = check_quality(output_file)

            if quality["valid"]:

                print("Fichier déjà présent et valide -> IGNORE")

                return {
                    "cluster": cluster,
                    "status": "DEJA_OK",
                    "n_images": np.nan,
                    "width": quality["width"],
                    "height": quality["height"],
                    "bands": quality["bands"],
                    "pct_non_zero": quality["pct_non_zero"],
                    "mean_reflectance":
                        quality["mean_reflectance"],
                    "error": ""
                }

            else:

                print(
                    "Fichier existant invalide -> "
                    "nouveau téléchargement"
                )

                output_file.unlink(
                    missing_ok=True
                )

        # --------------------------------------------------------
        # Région de 5 km x 5 km
        # --------------------------------------------------------

        point = ee.Geometry.Point(
            [lon, lat]
        )

        region = (
            point
            .buffer(PATCH_RADIUS_METERS)
            .bounds()
        )

        composite, n_images = (
            create_local_composite(region)
        )

        print(
            "Images Sentinel-2 :",
            n_images
        )

        if composite is None:

            return {
                "cluster": cluster,
                "status": "ECHEC_AUCUNE_IMAGE",
                "n_images": 0,
                "width": 0,
                "height": 0,
                "bands": 0,
                "pct_non_zero": 0.0,
                "mean_reflectance": 0.0,
                "error": "Aucune image Sentinel-2"
            }

        # --------------------------------------------------------
        # URL EARTH ENGINE
        # --------------------------------------------------------

        url = composite.getDownloadURL(
            {
                "region": region,
                "scale": SCALE_METERS,
                "crs": "EPSG:4326",
                "format": "GEO_TIFF"
            }
        )

        response = requests.get(
            url,
            timeout=180
        )

        response.raise_for_status()

        # --------------------------------------------------------
        # ECRITURE DU GEOTIFF
        # --------------------------------------------------------

        if response.content[:2] == b"PK":

            with zipfile.ZipFile(
                io.BytesIO(response.content)
            ) as archive:

                tif_files = [
                    name
                    for name in archive.namelist()
                    if name.lower().endswith(".tif")
                ]

                if not tif_files:

                    raise RuntimeError(
                        "Aucun GeoTIFF reçu."
                    )

                with archive.open(
                    tif_files[0]
                ) as source:

                    output_file.write_bytes(
                        source.read()
                    )

        else:

            output_file.write_bytes(
                response.content
            )

        # --------------------------------------------------------
        # CONTROLE
        # --------------------------------------------------------

        quality = check_quality(
            output_file
        )

        print(
            f"Dimensions : "
            f"{quality['width']} x "
            f"{quality['height']}"
        )

        print(
            f"Pixels non nuls : "
            f"{quality['pct_non_zero']:.2f} %"
        )

        print(
            f"Réflectance moyenne : "
            f"{quality['mean_reflectance']:.4f}"
        )

        if quality["valid"]:

            print("QUALITE : OK")
            status = "OK"

        else:

            print("QUALITE : ECHEC")
            status = "ECHEC_QUALITE"

        return {
            "cluster": cluster,
            "status": status,
            "n_images": n_images,
            "width": quality["width"],
            "height": quality["height"],
            "bands": quality["bands"],
            "pct_non_zero": quality["pct_non_zero"],
            "mean_reflectance":
                quality["mean_reflectance"],
            "error": ""
        }


    # ============================================================
    # 8. TRAITEMENT DES 539 CLUSTERS
    # ============================================================

    print("\n====================================")
    print("EXTRACTION SENTINEL-2")
    print("PATCH : ENVIRON 5 KM x 5 KM")
    print("TOTAL :", len(df), "CLUSTERS")
    print("====================================")


    results = []

    total = len(df)


    for position, (_, row) in enumerate(
        df.iterrows(),
        start=1
    ):

        cluster = int(
            row["cluster"]
        )

        print(
            f"\n[{position}/{total}] "
            f"Traitement du cluster {cluster}"
        )

        try:

            result = process_cluster(
                row
            )

        except Exception as error:

            print(
                "ERREUR :",
                str(error)
            )

            result = {
                "cluster": cluster,
                "status": "ERREUR",
                "n_images": np.nan,
                "width": 0,
                "height": 0,
                "bands": 0,
                "pct_non_zero": 0.0,
                "mean_reflectance": 0.0,
                "error": str(error)
            }

        results.append(
            result
        )

        # Sauvegarde après CHAQUE cluster
        # -> permet de garder la trace même en cas
        #    d'interruption du programme.
        save_log(results)

        # Petite pause pour éviter d'enchaîner
        # trop brutalement les requêtes.
        time.sleep(0.5)


    # ============================================================
    # 9. BILAN FINAL
    # ============================================================

    results_df = pd.DataFrame(
        results
    )

    successful = results_df[
        results_df["status"].isin(
            ["OK", "DEJA_OK"]
        )
    ]

    problems = results_df[
        ~results_df["status"].isin(
            ["OK", "DEJA_OK"]
        )
    ]


    print("\n")
    print("====================================")
    print("BILAN FINAL")
    print("====================================")

    print(
        "Nombre total :",
        len(results_df)
    )

    print(
        "Images valides :",
        len(successful)
    )

    print(
        "Problèmes :",
        len(problems)
    )

    if len(problems) > 0:

        print("\nClusters à vérifier :")

        print(
            problems[
                [
                    "cluster",
                    "status",
                    "error"
                ]
            ].to_string(
                index=False
            )
        )


    print("\nJournal sauvegardé :")
    print(LOG_FILE)

    print("\nImages sauvegardées :")
    print(OUTPUT_DIR)

    print("\n====================================")

    if len(successful) == total:

        print(
            "EXTRACTION COMPLETE : "
            "539/539 CLUSTERS VALIDES"
        )

    else:

        print(
            "EXTRACTION TERMINEE AVEC "
            f"{len(problems)} PROBLEME(S)"
        )

    print("====================================")

if __name__ == "__main__":
    main()
