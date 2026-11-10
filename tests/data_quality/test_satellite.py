from pathlib import Path

import numpy as np
import rasterio


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SATELLITE_DIR = PROJECT_ROOT / "data" / "raw" / "satellite"


def get_images():
    """Retourne toutes les images GeoTIFF Sentinel-2."""
    return sorted(SATELLITE_DIR.glob("*.tif"))


def test_dossier_satellite_existe():
    """Le dossier contenant les images doit exister."""
    assert SATELLITE_DIR.exists()


def test_nombre_images():
    """Une image doit être disponible pour chacun des 539 clusters."""
    images = get_images()
    assert len(images) == 539, (
        f"539 images attendues, {len(images)} trouvées."
    )


def test_images_lisibles():
    """Toutes les images doivent pouvoir être ouvertes."""
    for image_path in get_images():
        with rasterio.open(image_path) as src:
            assert src.width > 0
            assert src.height > 0


def test_nombre_bandes():
    """Chaque image RGB doit contenir exactement trois bandes."""
    for image_path in get_images():
        with rasterio.open(image_path) as src:
            assert src.count == 3, (
                f"{image_path.name} contient {src.count} bandes."
            )


def test_dimensions_images():
    """Les images doivent avoir des dimensions cohérentes avec les patchs 5 km."""
    for image_path in get_images():
        with rasterio.open(image_path) as src:
            assert src.width >= 450
            assert src.height >= 450


def test_pixels_non_vides():
    """Les images ne doivent pas être entièrement vides."""
    for image_path in get_images():
        with rasterio.open(image_path) as src:
            data = src.read()

        assert np.any(data != 0), (
            f"{image_path.name} est entièrement vide."
        )


def test_proportion_pixels_valides():
    """Au moins 90 % des pixels doivent être non nuls."""
    for image_path in get_images():
        with rasterio.open(image_path) as src:
            data = src.read()

        proportion = np.count_nonzero(data) / data.size

        assert proportion >= 0.90, (
            f"{image_path.name}: seulement "
            f"{proportion:.2%} de pixels non nuls."
        )


def test_reflectance_positive():
    """La réflectance moyenne doit être strictement positive."""
    for image_path in get_images():
        with rasterio.open(image_path) as src:
            data = src.read().astype("float32")

        assert np.mean(data) > 0, (
            f"Réflectance anormale pour {image_path.name}."
        )