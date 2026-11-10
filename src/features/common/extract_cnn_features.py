"""Extraction des caractéristiques CNN pour une zone d'images.

Usage :
    python src/features/common/extract_cnn_features.py --zone dhs --extractor imagenet --tiles
    python src/features/common/extract_cnn_features.py --zone grid --extractor national --tiles

Zones : dhs (539 grappes), ehcvm (grappes EHCVM), grid (3 218 points nationaux).
Extracteurs : imagenet (ResNet-50 pré-entraîné) ou national (ResNet-50
adapté aux classes VIIRS, étape common/12).

--tiles : chaque image de 5 km est découpée en tuiles d'environ 1 km
(config.yaml : features.tile_size_px), chaque tuile passe dans le CNN et les
vecteurs sont moyennés. C'est l'agrégation de Xie et al. (2016) ; sans
--tiles, l'image entière est redimensionnée en 224 px (étapes 06 et 13),
ce qui dégrade la résolution d'environ 2,2 fois.

Sortie : voir src/utils/feature_sets.py ; l'extraction reprend là où elle
s'est arrêtée (identifiants déjà présents ignorés).
"""
from pathlib import Path
import argparse
import sys

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd

from src.utils.cnn import N_FEATURES, build_extractor, extract_mean_features
from src.utils.config import load_config
from src.utils.feature_sets import IMAGE_SOURCES, cnn_file
from src.utils.imagery import image_id, read_rgb, split_tiles
from src.utils.paths import COMMON_MODELS_DIR, RAW_DIR

NATIONAL_CHECKPOINT = COMMON_MODELS_DIR / "resnet50_national_viirs_finetuned.pt"
SAVE_EVERY = 50


def append_rows(rows, output):
    """Ajouter les lignes au fichier (sauvegarde progressive) ; renvoie une liste vide."""
    if rows:
        output.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(output, mode="a", header=not output.exists(), index=False)
    return []


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--zone", choices=IMAGE_SOURCES, required=True)
    parser.add_argument("--extractor", choices=["imagenet", "national"], required=True)
    parser.add_argument("--tiles", action="store_true", help="Moyenne de tuiles d'environ 1 km")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    config = load_config()
    folder, prefix, id_column = IMAGE_SOURCES[args.zone]
    images = sorted((RAW_DIR / folder).glob("*.tif"))
    name = f"cnn_{args.extractor}" + ("_tiles" if args.tiles else "")
    output = cnn_file(args.zone, name)
    tile_px = config["features"]["tile_size_px"] if args.tiles else 0
    done = set(pd.read_csv(output, usecols=[id_column])[id_column]) if output.exists() else set()
    todo = [p for p in images if image_id(p, prefix) not in done]
    print(f"{len(images)} images | déjà extraites : {len(done)} | à traiter : {len(todo)} | sortie : {output}")
    if args.dry_run:
        return
    if not images:
        raise FileNotFoundError(f"Aucune image dans {RAW_DIR / folder}")

    model, transform = build_extractor(args.extractor, NATIONAL_CHECKPOINT)
    rows, skipped = [], []
    max_reflectance = config["satellite"]["max_reflectance"]
    for position, path in enumerate(todo, start=1):
        tiles = split_tiles(read_rgb(path, max_reflectance), tile_px)
        if not tiles:
            skipped.append(path.name)
            continue
        vector = extract_mean_features(model, transform, tiles, config["features"]["batch_size"])
        rows.append({id_column: image_id(path, prefix), "n_tiles": len(tiles),
                     **{f"cnn_{j:04d}": float(v) for j, v in enumerate(vector)}})
        if len(rows) == SAVE_EVERY:
            rows = append_rows(rows, output)
            print(f"{position}/{len(todo)} images traitées")
    append_rows(rows, output)
    if skipped:
        print(f"Images sans tuile valide (ignorées) : {skipped}")
    result = pd.read_csv(output)
    if result[id_column].duplicated().any():
        raise ValueError("Identifiants dupliqués dans la sortie.")
    if result.filter(like="cnn_").shape[1] != N_FEATURES:
        raise ValueError("Nombre de caractéristiques inattendu.")
    print(f"Terminé : {len(result)} lignes dans {output}")


if __name__ == "__main__":
    main()
