# Résolution des chemins indépendante du répertoire de lancement.
from pathlib import Path
import sys

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.utils.paths import (
    DHS_MODELS_DIR,
    DHS_PROCESSED_DIR,
    DHS_SATELLITE_DIR,
    PROJECT_ROOT,
    ensure_layout,
)

ensure_layout('dhs')

import re

import numpy as np
import pandas as pd
import rasterio
import torch
import torch.nn as nn

from PIL import Image
from torchvision.models import resnet50, ResNet50_Weights


# ============================================================
# 06D - EXTRACTION DES FEATURES RESNET-50 ADAPTE A VIIRS
# ============================================================


SATELLITE_DIR = (
    DHS_SATELLITE_DIR
)

MODEL_FILE = (
    DHS_MODELS_DIR / 'resnet50_viirs_finetuned.pt'
)

OUTPUT_FILE = (
    DHS_PROCESSED_DIR / 'cnn_features_resnet50_viirs.csv'
)

ORIGINAL_FEATURES_FILE = (
    DHS_PROCESSED_DIR / 'cnn_features_resnet50.csv'
)

EXPECTED_CLUSTERS = 539
EXPECTED_FEATURES = 2048
BATCH_SIZE = 16


# ============================================================
# 1. DEVICE
# ============================================================

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

print("=" * 70)
print("EXTRACTION DES FEATURES CNN APRES ADAPTATION VIIRS")
print("=" * 70)
print(f"Device utilisé : {DEVICE}")


# ============================================================
# 2. IDENTIFICATION DES CLUSTERS
# ============================================================

def extract_cluster_id(path):

    match = re.search(
        r"cluster_(\d+)",
        path.stem
    )

    if match is None:
        raise ValueError(
            f"Cluster non identifiable : {path.name}"
        )

    return int(match.group(1))


# ============================================================
# 3. PRETRAITEMENT
# ============================================================

weights = ResNet50_Weights.DEFAULT
preprocess = weights.transforms()


def load_satellite_image(path):

    with rasterio.open(path) as src:
        image = src.read().astype(np.float32)

    image = np.transpose(
        image,
        (1, 2, 0)
    )

    image = np.nan_to_num(
        image,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    # Même transformation que 06 / 06B / 06C
    image = np.clip(
        image,
        0.0,
        0.30
    )

    image = image / 0.30

    image = (
        image * 255
    ).astype(np.uint8)

    pil_image = Image.fromarray(
        image,
        mode="RGB"
    )

    return preprocess(pil_image)


# ============================================================
# 4. RECONSTRUCTION DU MODELE 06C
# ============================================================

model = resnet50(
    weights=None
)

n_features = model.fc.in_features

model.fc = nn.Sequential(
    nn.Linear(n_features, 256),
    nn.ReLU(),
    nn.Dropout(p=0.30),
    nn.Linear(256, 1)
)


# ============================================================
# 5. CHARGEMENT DU CHECKPOINT FINE-TUNE
# ============================================================

if not MODEL_FILE.exists():
    raise FileNotFoundError(
        f"Modèle introuvable : {MODEL_FILE}"
    )

checkpoint = torch.load(
    MODEL_FILE,
    map_location="cpu",
    weights_only=False
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

print("Checkpoint fine-tuné chargé : OK")


# ============================================================
# 6. SUPPRESSION DE LA TETE VIIRS
# ============================================================

model.fc = nn.Identity()

model = model.to(DEVICE)
model.eval()

print("Tête VIIRS retirée : OK")
print("Dimension attendue : 2048 features")


# ============================================================
# 7. IMAGES
# ============================================================

image_paths = sorted(
    SATELLITE_DIR.glob("*.tif"),
    key=extract_cluster_id
)

print(
    f"Nombre d'images trouvées : {len(image_paths)}"
)

if len(image_paths) != EXPECTED_CLUSTERS:
    raise ValueError(
        f"{EXPECTED_CLUSTERS} images attendues, "
        f"{len(image_paths)} trouvées."
    )

cluster_ids = [
    extract_cluster_id(path)
    for path in image_paths
]

if len(set(cluster_ids)) != EXPECTED_CLUSTERS:
    raise ValueError(
        "Identifiants de clusters dupliqués."
    )


# ============================================================
# 8. EXTRACTION
# ============================================================

rows = []

with torch.no_grad():

    for start in range(
        0,
        len(image_paths),
        BATCH_SIZE
    ):

        batch_paths = image_paths[
            start:start + BATCH_SIZE
        ]

        tensors = []
        batch_ids = []

        for path in batch_paths:

            tensors.append(
                load_satellite_image(path)
            )

            batch_ids.append(
                extract_cluster_id(path)
            )

        batch = torch.stack(
            tensors
        ).to(DEVICE)

        features = model(batch)

        features = (
            features
            .cpu()
            .numpy()
        )

        for cluster_id, vector in zip(
            batch_ids,
            features
        ):

            row = {
                "cluster": cluster_id
            }

            for j, value in enumerate(vector):

                row[
                    f"cnn_viirs_{j:04d}"
                ] = float(value)

            rows.append(row)

        current = min(
            start + BATCH_SIZE,
            len(image_paths)
        )

        print(
            f"Clusters traités : "
            f"{current}/{len(image_paths)}"
        )


# ============================================================
# 9. BASE FINALE
# ============================================================

features_df = pd.DataFrame(rows)

features_df = (
    features_df
    .sort_values("cluster")
    .reset_index(drop=True)
)

n_clusters = len(features_df)

n_features_final = (
    features_df.shape[1] - 1
)

n_missing = int(
    features_df.isna().sum().sum()
)

n_duplicates = int(
    features_df["cluster"]
    .duplicated()
    .sum()
)


print()
print("=" * 70)
print("CONTROLE DES NOUVELLES FEATURES")
print("=" * 70)

print(
    f"Clusters : {n_clusters}"
)

print(
    f"Features : {n_features_final}"
)

print(
    f"Valeurs manquantes : {n_missing}"
)

print(
    f"Doublons : {n_duplicates}"
)


if n_clusters != EXPECTED_CLUSTERS:
    raise ValueError(
        "Nombre incorrect de clusters."
    )

if n_features_final != EXPECTED_FEATURES:
    raise ValueError(
        "Nombre incorrect de features."
    )

if n_missing != 0:
    raise ValueError(
        "Valeurs manquantes détectées."
    )

if n_duplicates != 0:
    raise ValueError(
        "Clusters dupliqués détectés."
    )


feature_columns = [
    c
    for c in features_df.columns
    if c.startswith("cnn_viirs_")
]

array = (
    features_df[
        feature_columns
    ]
    .to_numpy()
)

if not np.isfinite(array).all():
    raise ValueError(
        "Valeurs infinies détectées."
    )

variances = (
    features_df[
        feature_columns
    ]
    .var()
)

n_variable = int(
    (variances > 0).sum()
)

print(
    f"Features avec variance > 0 : "
    f"{n_variable}/{EXPECTED_FEATURES}"
)


# ============================================================
# 10. COMPARAISON AVEC LES FEATURES IMAGENET
# ============================================================

if ORIGINAL_FEATURES_FILE.exists():

    print()
    print("=" * 70)
    print("COMPARAISON IMAGENET VS VIIRS")
    print("=" * 70)

    original = pd.read_csv(
        ORIGINAL_FEATURES_FILE
    )

    original = (
        original
        .sort_values("cluster")
        .reset_index(drop=True)
    )

    if not np.array_equal(
        original["cluster"].to_numpy(),
        features_df["cluster"].to_numpy()
    ):
        raise ValueError(
            "Les clusters des deux fichiers "
            "ne correspondent pas."
        )

    original_columns = [
        c
        for c in original.columns
        if c.startswith("cnn_")
    ]

    original_array = (
        original[
            original_columns
        ]
        .to_numpy()
    )

    if original_array.shape != array.shape:
        raise ValueError(
            "Dimensions incompatibles entre "
            "les anciennes et nouvelles features."
        )

    mean_absolute_difference = float(
        np.mean(
            np.abs(
                array - original_array
            )
        )
    )

    mean_squared_difference = float(
        np.mean(
            (
                array - original_array
            ) ** 2
        )
    )

    identical = bool(
        np.allclose(
            array,
            original_array
        )
    )

    print(
        f"Différence absolue moyenne : "
        f"{mean_absolute_difference:.6f}"
    )

    print(
        f"Différence quadratique moyenne : "
        f"{mean_squared_difference:.6f}"
    )

    print(
        f"Features identiques : {identical}"
    )

else:

    print(
        "\nAncien fichier de features absent : "
        "comparaison ignorée."
    )


# ============================================================
# 11. SAUVEGARDE
# ============================================================

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

features_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print()
print(
    f"Fichier sauvegardé : {OUTPUT_FILE}"
)

print()
print("=" * 70)
print("EXTRACTION DES FEATURES ADAPTEES TERMINEE")
print("=" * 70)