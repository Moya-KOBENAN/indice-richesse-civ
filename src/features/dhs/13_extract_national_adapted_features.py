"""
ETAPE 13
Extraction des features ResNet-50 adaptées nationalement à VIIRS
sur les 539 images Sentinel-2 des grappes DHS.

Entrée :
    - data/raw/satellite/
    - outputs/common/models/resnet50_national_viirs_finetuned.pt

Sortie :
    - data/processed/dhs/cnn_features_resnet50_viirs_national.csv

Le réseau a été adapté au proxy VIIRS sur la grille nationale
indépendante de 3218 points. La tête de classification VIIRS est
retirée afin de récupérer le vecteur de 2048 features produit par
le backbone ResNet-50.
"""


# Résolution des chemins indépendante du répertoire de lancement.
from pathlib import Path
import sys

_PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.utils.paths import (
    COMMON_MODELS_DIR,
    DHS_PROCESSED_DIR,
    DHS_SATELLITE_DIR,
    PROJECT_ROOT,
    ensure_layout,
)



import re

# ---------------------------------------------------------------------
# Racine du projet
# ---------------------------------------------------------------------


import numpy as np
import pandas as pd
import rasterio

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from PIL import Image

from torchvision.models import resnet50, ResNet50_Weights


# ---------------------------------------------------------------------
# Paramètres
# ---------------------------------------------------------------------
MODEL_PATH = (
    COMMON_MODELS_DIR
    / "resnet50_national_viirs_finetuned.pt"
)

OUTPUT_PATH = (
    DHS_PROCESSED_DIR
    / "cnn_features_resnet50_viirs_national.csv"
)

EXPECTED_CLUSTERS = 539
EXPECTED_FEATURES = 2048
NUM_CLASSES = 5
BATCH_SIZE = 16


# ---------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


# ---------------------------------------------------------------------
# Lecture et préparation Sentinel-2
# ---------------------------------------------------------------------
def read_satellite_image(path):
    """
    Reproduit la préparation utilisée dans les étapes CNN précédentes :
    - lecture RGB Sentinel-2
    - nettoyage NaN/inf
    - réflectance limitée à [0, 0.30]
    - conversion en image RGB 8 bits
    """

    with rasterio.open(path) as src:
        image = src.read()

    if image.shape[0] != 3:
        raise ValueError(
            f"{path.name} : {image.shape[0]} bandes au lieu de 3."
        )

    image = np.transpose(image, (1, 2, 0))

    image = np.nan_to_num(
        image,
        nan=0.0,
        posinf=0.30,
        neginf=0.0,
    )

    image = np.clip(image, 0.0, 0.30)
    image = image / 0.30
    image = (image * 255.0).clip(0, 255).astype(np.uint8)

    return Image.fromarray(image, mode="RGB")


# ---------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------
class DHSSatelliteDataset(Dataset):

    def __init__(self, records, transform):
        self.records = records
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):

        cluster_id, image_path = self.records[index]

        image = read_satellite_image(image_path)
        image = self.transform(image)

        return cluster_id, image


# ---------------------------------------------------------------------
# Recherche des images DHS
# ---------------------------------------------------------------------
def get_dhs_images():

    files = sorted(DHS_SATELLITE_DIR.glob("*.tif"))

    records = []

    for path in files:

        match = re.search(
            r"cluster_(\d+)",
            path.stem
        )

        if match:
            cluster_id = int(match.group(1))
            records.append(
                (cluster_id, path)
            )

    records = sorted(
        records,
        key=lambda x: x[0]
    )

    ids = [x[0] for x in records]

    if len(records) != EXPECTED_CLUSTERS:
        raise ValueError(
            f"{len(records)} images reconnues au lieu de "
            f"{EXPECTED_CLUSTERS}."
        )

    if len(set(ids)) != EXPECTED_CLUSTERS:
        raise ValueError(
            "Les identifiants de grappes ne sont pas uniques."
        )

    return records


# ---------------------------------------------------------------------
# Construction du modèle national
# ---------------------------------------------------------------------
def load_national_model():

    print("Chargement du checkpoint national...")

    checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=False,
    )

    if "model_state_dict" not in checkpoint:
        raise KeyError(
            "model_state_dict absent du checkpoint."
        )

    num_classes = checkpoint.get(
        "num_classes",
        NUM_CLASSES
    )

    print(
        "Phase checkpoint :",
        checkpoint.get("phase", "non renseignée")
    )

    print(
        "Balanced accuracy sauvegardée :",
        checkpoint.get(
            "best_balanced_accuracy",
            "non renseignée"
        )
    )

    print(
        "Nombre de classes :",
        num_classes
    )

    # Architecture identique à l'étape 12
    model = resnet50(weights=None)

    model.fc = nn.Sequential(
        nn.Linear(2048, 256),
        nn.ReLU(),
        nn.Dropout(0.30),
        nn.Linear(256, num_classes),
    )

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    print("Checkpoint chargé avec succès.")

    # Suppression de la tête VIIRS.
    # La sortie devient le vecteur de 2048 features.
    model.fc = nn.Identity()

    model = model.to(DEVICE)
    model.eval()

    return model


# ---------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------
def extract_features(
    model,
    loader
):

    all_clusters = []
    all_features = []

    with torch.no_grad():

        for batch_index, (
            cluster_ids,
            images
        ) in enumerate(loader, start=1):

            images = images.to(DEVICE)

            features = model(images)

            features = (
                features
                .detach()
                .cpu()
                .numpy()
            )

            all_clusters.extend(
                cluster_ids.numpy().tolist()
            )

            all_features.append(
                features
            )

            processed = min(
                batch_index * BATCH_SIZE,
                len(loader.dataset)
            )

            print(
                f"\rExtraction : "
                f"{processed}/{len(loader.dataset)}",
                end="",
                flush=True,
            )

    print()

    feature_matrix = np.vstack(
        all_features
    )

    return (
        np.array(all_clusters),
        feature_matrix,
    )


# ---------------------------------------------------------------------
# Validation finale
# ---------------------------------------------------------------------
def validate_output(df):

    feature_cols = [
        c for c in df.columns
        if c.startswith("feature_")
    ]

    print()
    print("=" * 72)
    print("CONTROLE DES FEATURES")
    print("=" * 72)

    print(
        "Nombre de grappes :",
        len(df)
    )

    print(
        "Nombre de features :",
        len(feature_cols)
    )

    print(
        "Clusters uniques :",
        df["cluster"].nunique()
    )

    print(
        "Valeurs manquantes :",
        int(df.isna().sum().sum())
    )

    print(
        "Doublons cluster :",
        int(df["cluster"].duplicated().sum())
    )

    X = df[feature_cols].to_numpy()

    variances = np.var(
        X,
        axis=0
    )

    zero_variance = int(
        np.sum(variances == 0)
    )

    print(
        "Features à variance nulle :",
        zero_variance
    )

    if len(df) != EXPECTED_CLUSTERS:
        raise ValueError(
            "Nombre de grappes incorrect."
        )

    if len(feature_cols) != EXPECTED_FEATURES:
        raise ValueError(
            "Nombre de features incorrect."
        )

    if df["cluster"].nunique() != EXPECTED_CLUSTERS:
        raise ValueError(
            "Clusters non uniques."
        )

    if df.isna().any().any():
        raise ValueError(
            "Valeurs manquantes détectées."
        )

    if X.shape != (
        EXPECTED_CLUSTERS,
        EXPECTED_FEATURES
    ):
        raise ValueError(
            f"Matrice incorrecte : {X.shape}"
        )


# ---------------------------------------------------------------------
# Programme principal
# ---------------------------------------------------------------------
def main():
    ensure_layout("dhs")

    print("=" * 72)
    print(
        "ETAPE 13 - FEATURES RESNET-50 NATIONAL -> DHS"
    )
    print("=" * 72)

    print(
        "Device :",
        DEVICE
    )

    print(
        "Images DHS :",
        DHS_SATELLITE_DIR
    )

    print(
        "Modèle national :",
        MODEL_PATH
    )

    print(
        "Sortie :",
        OUTPUT_PATH
    )

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Modèle introuvable : {MODEL_PATH}"
        )

    records = get_dhs_images()

    print()
    print(
        "Images DHS reconnues :",
        len(records)
    )

    print(
        "Premier cluster :",
        records[0][0]
    )

    print(
        "Dernier cluster :",
        records[-1][0]
    )

    # Transformation ImageNet utilisée par ResNet-50
    weights = ResNet50_Weights.DEFAULT
    transform = weights.transforms()

    dataset = DHSSatelliteDataset(
        records,
        transform
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    model = load_national_model()

    print()
    print(
        "Extraction des 2048 features..."
    )

    cluster_ids, features = extract_features(
        model,
        loader
    )

    print(
        "Dimension matrice :",
        features.shape
    )

    columns = [
        f"feature_{i:04d}"
        for i in range(
            1,
            EXPECTED_FEATURES + 1
        )
    ]

    df = pd.DataFrame(
        features,
        columns=columns
    )

    df.insert(
        0,
        "cluster",
        cluster_ids
    )

    df = df.sort_values(
        "cluster"
    ).reset_index(drop=True)

    validate_output(df)

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print()
    print(
        "Fichier sauvegardé :"
    )

    print(
        OUTPUT_PATH
    )

    print()
    print("=" * 72)
    print(
        "ETAPE 13 TERMINEE"
    )
    print("=" * 72)


if __name__ == "__main__":
    main()
