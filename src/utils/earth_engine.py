"""Fonctions Earth Engine partagées : composites Sentinel-2 et statistiques VIIRS.

Logique identique à l'étape DHS 05 (masque SCL, médiane annuelle, seuil de
nuages avec repli) et au script GEE d'extraction VIIRS DHS, afin que les
variables EHCVM soient construites comme celles de la DHS.
"""
import io
import zipfile

import numpy as np

# Classes SCL exclues : pas de données, saturé, ombre de nuage, nuages, cirrus, neige
_SCL_EXCLUDED = (0, 1, 3, 8, 9, 10, 11)


def initialize(project_id):
    import ee

    ee.Initialize(project=project_id)
    return ee


def _mask_s2(image, bands):
    scl = image.select("SCL")
    mask = scl.neq(_SCL_EXCLUDED[0])
    for value in _SCL_EXCLUDED[1:]:
        mask = mask.And(scl.neq(value))
    return image.updateMask(mask).select(bands).divide(10000)


def sentinel2_composite(ee, region, sat):
    """Composite médian ; seuil de nuages relâché si trop peu d'images."""
    def collection(threshold):
        return (ee.ImageCollection(sat["collection"])
                .filterDate(sat["start_date"], sat["end_date"])
                .filterBounds(region)
                .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", threshold)))

    images = collection(sat["cloud_threshold"])
    n_images = images.size().getInfo()
    if n_images < sat["minimum_images"]:
        images = collection(sat["fallback_cloud_threshold"])
        n_images = images.size().getInfo()
    if n_images == 0:
        return None, 0
    return images.map(lambda im: _mask_s2(im, sat["bands"])).median(), n_images


def download_patch(ee, composite, region, scale_m, output_file):
    """Télécharger un GeoTIFF (éventuellement zippé par Earth Engine)."""
    import requests

    url = composite.getDownloadURL({"region": region, "scale": scale_m,
                                    "crs": "EPSG:4326", "format": "GEO_TIFF"})
    response = requests.get(url, timeout=180)
    response.raise_for_status()
    content = response.content
    if content[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            tifs = [n for n in archive.namelist() if n.lower().endswith(".tif")]
            if not tifs:
                raise RuntimeError("Aucun GeoTIFF dans la réponse Earth Engine.")
            content = archive.read(tifs[0])
    output_file.write_bytes(content)


def patch_quality(path, min_size_px=450, min_nonzero_ratio=0.90):
    """Contrôle d'une image téléchargée : 3 bandes, taille, pixels non nuls."""
    import rasterio

    try:
        with rasterio.open(path) as source:
            data = source.read(masked=True)
            values = data.compressed()
            nonzero = np.count_nonzero(values) / len(values) if len(values) else 0.0
            valid = (source.count == 3 and source.width >= min_size_px
                     and source.height >= min_size_px and nonzero >= min_nonzero_ratio)
            return {"valid": bool(valid), "width": source.width, "height": source.height,
                    "bands": source.count, "pct_non_zero": 100 * nonzero,
                    "mean_reflectance": float(values.mean()) if len(values) else 0.0}
    except Exception as error:  # fichier corrompu : à retélécharger
        return {"valid": False, "width": 0, "height": 0, "bands": 0,
                "pct_non_zero": 0.0, "mean_reflectance": 0.0, "error": str(error)}


def viirs_buffer_stats(ee, points, viirs_cfg, radius_cfg):
    """Statistiques VIIRS dans trois tampons, comme le script GEE DHS.

    `points` : ee.FeatureCollection avec une propriété `urban_rural` (U/R).
    """
    image = (ee.ImageCollection(viirs_cfg["collection"])
             .filterDate(f"{viirs_cfg['year']}-01-01", f"{viirs_cfg['year'] + 1}-01-01")
             .first())
    radiance = image.select("average_masked").rename("viirs_rad")
    radiance = radiance.updateMask(radiance.gte(0))
    coverage = image.select("cf_cvg").rename("viirs_cf_cvg")
    reducer = (ee.Reducer.mean().combine(ee.Reducer.median(), "", True)
               .combine(ee.Reducer.max(), "", True).combine(ee.Reducer.stdDev(), "", True))
    scale = radius_cfg["scale_m"]

    def stats(feature, radius, prefix):
        zone = feature.geometry().buffer(radius)
        rad = radiance.reduceRegion(reducer=reducer, geometry=zone, scale=scale,
                                    bestEffort=True, maxPixels=1e8)
        cov = coverage.reduceRegion(reducer=ee.Reducer.mean(), geometry=zone, scale=scale,
                                    bestEffort=True, maxPixels=1e8)
        return (feature.set(f"{prefix}_radius_m", radius)
                .set(f"{prefix}_mean", rad.get("viirs_rad_mean"))
                .set(f"{prefix}_median", rad.get("viirs_rad_median"))
                .set(f"{prefix}_max", rad.get("viirs_rad_max"))
                .set(f"{prefix}_stddev", rad.get("viirs_rad_stdDev"))
                .set(f"{prefix}_coverage", cov.get("viirs_cf_cvg")))

    def per_point(feature):
        urban = ee.String(feature.get("urban_rural")).equals("U")
        out = stats(feature, ee.Number(ee.Algorithms.If(
            urban, radius_cfg["urban_radius_m"], radius_cfg["rural_radius_m"])), "viirs_2u5r")
        out = stats(out, radius_cfg["all_radius_m"], "viirs_5all")
        return stats(out, ee.Number(ee.Algorithms.If(
            urban, radius_cfg["urban_radius_m"], radius_cfg["rural_radius_sensitivity_m"])), "viirs_2u10r")

    return points.map(per_point)
