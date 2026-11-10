"""Lecture des images Sentinel-2 et découpage en tuiles.

La conversion en RGB 8 bits reprend exactement celle des étapes 06, 12 et 13
(réflectance bornée à [0, max_reflectance]) : les caractéristiques extraites
restent comparables entre enquêtes et avec la grille nationale.
"""
import re

import numpy as np


def read_rgb(path, max_reflectance=0.30):
    """Image (H, W, 3) uint8 à partir d'un GeoTIFF B4-B3-B2 en réflectance."""
    import rasterio

    with rasterio.open(path) as source:
        image = source.read().astype(np.float32)
    if image.shape[0] != 3:
        raise ValueError(f"{path.name} : {image.shape[0]} bandes au lieu de 3.")
    image = np.transpose(image, (1, 2, 0))
    image = np.nan_to_num(image, nan=0.0, posinf=max_reflectance, neginf=0.0)
    image = np.clip(image, 0.0, max_reflectance) / max_reflectance
    return (image * 255.0).round().astype(np.uint8)


def split_tiles(image, tile_px, min_valid_ratio=0.5):
    """Tuiles carrées de `tile_px` pixels ; les bords incomplets sont ignorés.

    Une tuile majoritairement vide (pixels nuls, masque nuageux) est écartée
    pour ne pas diluer la moyenne des caractéristiques.
    """
    if tile_px <= 0:
        return [image]
    height, width = image.shape[:2]
    tiles = []
    for top in range(0, height - tile_px + 1, tile_px):
        for left in range(0, width - tile_px + 1, tile_px):
            tile = image[top:top + tile_px, left:left + tile_px]
            if (tile.max(axis=2) > 0).mean() >= min_valid_ratio:
                tiles.append(tile)
    return tiles


def image_id(path, prefix):
    """Identifiant numérique dans un nom du type `<prefix>_0012_sentinel2_2021.tif`."""
    match = re.search(rf"{prefix}_(\d+)", path.stem)
    if match is None:
        raise ValueError(f"Identifiant « {prefix} » introuvable dans {path.name}")
    return int(match.group(1))
