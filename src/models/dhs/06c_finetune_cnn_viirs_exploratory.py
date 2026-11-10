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
    DHS_TABLES_DIR,
    PROJECT_ROOT,
    ensure_layout,
)

ensure_layout('dhs')

import copy
import random
import re

import numpy as np
import pandas as pd
import rasterio
import torch
import torch.nn as nn

from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.models import resnet50, ResNet50_Weights


# ============================================================
# 06C - FINE-TUNING RESNET-50 SUR LES LUMIERES NOCTURNES
# Projet : Indice de richesse - Côte d'Ivoire
#
# Phase 2 :
# - chargement du modèle obtenu en 06B
# - backbone globalement gelé
# - déblocage de layer4
# - fine-tuning avec un faible learning rate
# - sauvegarde du meilleur modèle
# ============================================================


# ============================================================
# 1. PARAMETRES
# ============================================================


SATELLITE_DIR = (
    DHS_SATELLITE_DIR
)

DATA_FILE = (
    DHS_PROCESSED_DIR / 'DHS_CI_2021_richesse_GPS_VIIRS_merged.csv'
)

INPUT_MODEL_FILE = (
    DHS_MODELS_DIR / 'resnet50_viirs_adapted.pt'
)

OUTPUT_MODEL_FILE = (
    DHS_MODELS_DIR / 'resnet50_viirs_finetuned.pt'
)

HISTORY_FILE = (
    DHS_TABLES_DIR / 'resnet50_viirs_finetuning_history.csv'
)


EXPECTED_CLUSTERS = 539

TARGET_COLUMN = "viirs_2u5r_mean"

BATCH_SIZE = 8

EPOCHS = 10

LEARNING_RATE_LAYER4 = 1e-5

LEARNING_RATE_HEAD = 1e-4

VALIDATION_RATIO = 0.20

RANDOM_STATE = 42


# ============================================================
# 2. REPRODUCTIBILITE
# ============================================================

random.seed(RANDOM_STATE)

np.random.seed(RANDOM_STATE)

torch.manual_seed(RANDOM_STATE)


# ============================================================
# 3. DEVICE
# ============================================================

if torch.backends.mps.is_available():

    DEVICE = torch.device("mps")

elif torch.cuda.is_available():

    DEVICE = torch.device("cuda")

else:

    DEVICE = torch.device("cpu")


print("=" * 72)
print("PHASE 2 - FINE-TUNING RESNET-50 SUR VIIRS")
print("=" * 72)

print(f"Device utilisé : {DEVICE}")


# ============================================================
# 4. IDENTIFICATION DU CLUSTER
# ============================================================

def extract_cluster_id(path):

    match = re.search(
        r"cluster_(\d+)",
        path.stem
    )

    if match is None:

        raise ValueError(
            f"Impossible d'identifier le cluster : {path.name}"
        )

    return int(
        match.group(1)
    )


# ============================================================
# 5. PRETRAITEMENT DES IMAGES
# ============================================================

weights = ResNet50_Weights.DEFAULT

preprocess = weights.transforms()


def load_satellite_image(path):

    with rasterio.open(path) as src:

        image = src.read().astype(
            np.float32
        )

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

    # Même transformation que le script 06
    image = np.clip(
        image,
        0.0,
        0.30
    )

    image = image / 0.30

    image = (
        image * 255
    ).astype(
        np.uint8
    )

    pil_image = Image.fromarray(
        image,
        mode="RGB"
    )

    return preprocess(
        pil_image
    )


# ============================================================
# 6. DONNEES
# ============================================================

df = pd.read_csv(
    DATA_FILE
)


if len(df) != EXPECTED_CLUSTERS:

    raise ValueError(
        f"{EXPECTED_CLUSTERS} clusters attendus, "
        f"{len(df)} trouvés."
    )


if TARGET_COLUMN not in df.columns:

    raise ValueError(
        f"Variable absente : {TARGET_COLUMN}"
    )


if df[TARGET_COLUMN].isna().any():

    raise ValueError(
        "Valeurs manquantes dans VIIRS."
    )


if (df[TARGET_COLUMN] < 0).any():

    raise ValueError(
        "Valeurs VIIRS négatives détectées."
    )


df["viirs_target_log"] = np.log1p(
    df[TARGET_COLUMN]
)


# ============================================================
# 7. ASSOCIATION AVEC LES IMAGES
# ============================================================

image_paths = sorted(
    SATELLITE_DIR.glob("*.tif"),
    key=extract_cluster_id
)


if len(image_paths) != EXPECTED_CLUSTERS:

    raise ValueError(
        f"{EXPECTED_CLUSTERS} images attendues, "
        f"{len(image_paths)} trouvées."
    )


image_map = {
    extract_cluster_id(path): path
    for path in image_paths
}


df["image_path"] = df["cluster"].map(
    image_map
)


if df["image_path"].isna().any():

    raise ValueError(
        "Certaines images Sentinel-2 sont manquantes."
    )


print(
    f"Clusters disponibles : {len(df)}"
)

print(
    f"Images associées : {df['image_path'].notna().sum()}"
)

print(
    f"Cible : log1p({TARGET_COLUMN})"
)


# ============================================================
# 8. DATASET
# ============================================================

class SatelliteVIIRSDataset(Dataset):

    def __init__(self, dataframe):

        self.dataframe = (
            dataframe
            .reset_index(drop=True)
        )

    def __len__(self):

        return len(
            self.dataframe
        )

    def __getitem__(self, index):

        row = self.dataframe.iloc[
            index
        ]

        image = load_satellite_image(
            Path(
                row["image_path"]
            )
        )

        target = torch.tensor(
            row["viirs_target_log"],
            dtype=torch.float32
        )

        cluster = int(
            row["cluster"]
        )

        return (
            image,
            target,
            cluster
        )


dataset = SatelliteVIIRSDataset(
    df
)


# ============================================================
# 9. MEME SPLIT QUE LA PHASE 06B
# ============================================================

n_total = len(dataset)

n_validation = int(
    round(
        n_total * VALIDATION_RATIO
    )
)

n_train = (
    n_total - n_validation
)


generator = torch.Generator()

generator.manual_seed(
    RANDOM_STATE
)


train_dataset, validation_dataset = random_split(
    dataset,
    [
        n_train,
        n_validation
    ],
    generator=generator
)


print(
    f"Train : {len(train_dataset)}"
)

print(
    f"Validation : {len(validation_dataset)}"
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)


validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ============================================================
# 10. RECONSTRUCTION DU MODELE
# ============================================================

model = resnet50(
    weights=weights
)


n_features = model.fc.in_features


model.fc = nn.Sequential(

    nn.Linear(
        n_features,
        256
    ),

    nn.ReLU(),

    nn.Dropout(
        p=0.30
    ),

    nn.Linear(
        256,
        1
    )
)


# ============================================================
# 11. CHARGEMENT DU MODELE 06B
# ============================================================

if not INPUT_MODEL_FILE.exists():

    raise FileNotFoundError(
        f"Modèle 06B introuvable : {INPUT_MODEL_FILE}"
    )


checkpoint = torch.load(
    INPUT_MODEL_FILE,
    map_location="cpu"
)


model.load_state_dict(
    checkpoint[
        "model_state_dict"
    ]
)


print(
    "Modèle 06B chargé : OK"
)


# ============================================================
# 12. GEL DU RESEAU
# ============================================================

for parameter in model.parameters():

    parameter.requires_grad = False


# ============================================================
# 13. DEBLOCAGE DE LAYER4
# ============================================================

for parameter in model.layer4.parameters():

    parameter.requires_grad = True


# La tête VIIRS reste également entraînable

for parameter in model.fc.parameters():

    parameter.requires_grad = True


model = model.to(
    DEVICE
)


# ============================================================
# 14. CONTROLE DES PARAMETRES
# ============================================================

total_parameters = sum(
    p.numel()
    for p in model.parameters()
)


trainable_parameters = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)


print(
    f"Paramètres totaux : {total_parameters:,}"
)

print(
    f"Paramètres entraînables : {trainable_parameters:,}"
)

print(
    "Bloc débloqué : layer4"
)


# ============================================================
# 15. OPTIMIZER
# ============================================================

optimizer = torch.optim.Adam(

    [
        {
            "params":
                model.layer4.parameters(),

            "lr":
                LEARNING_RATE_LAYER4,
        },

        {
            "params":
                model.fc.parameters(),

            "lr":
                LEARNING_RATE_HEAD,
        },
    ]
)


criterion = nn.MSELoss()


# ============================================================
# 16. EVALUATION
# ============================================================

def evaluate(loader):

    model.eval()

    predictions = []

    targets = []


    with torch.no_grad():

        for (
            images,
            target,
            _
        ) in loader:

            images = images.to(
                DEVICE
            )

            target = target.to(
                DEVICE
            )


            prediction = (
                model(images)
                .squeeze(1)
            )


            predictions.extend(
                prediction
                .cpu()
                .numpy()
            )

            targets.extend(
                target
                .cpu()
                .numpy()
            )


    predictions = np.asarray(
        predictions
    )

    targets = np.asarray(
        targets
    )


    mse = float(
        np.mean(
            (
                predictions
                - targets
            ) ** 2
        )
    )


    rmse = float(
        np.sqrt(
            mse
        )
    )


    mae = float(
        np.mean(
            np.abs(
                predictions
                - targets
            )
        )
    )


    return (
        mse,
        rmse,
        mae
    )


# ============================================================
# 17. FINE-TUNING
# ============================================================

print()
print("=" * 72)
print("DEBUT DU FINE-TUNING")
print("=" * 72)


history = []

best_validation_mse = float(
    "inf"
)

best_state = None


for epoch in range(
    1,
    EPOCHS + 1
):

    model.train()

    train_losses = []


    for (
        images,
        target,
        _
    ) in train_loader:

        images = images.to(
            DEVICE
        )

        target = target.to(
            DEVICE
        )


        optimizer.zero_grad()


        prediction = (
            model(images)
            .squeeze(1)
        )


        loss = criterion(
            prediction,
            target
        )


        loss.backward()

        optimizer.step()


        train_losses.append(
            loss.item()
        )


    train_mse = float(
        np.mean(
            train_losses
        )
    )


    (
        validation_mse,
        validation_rmse,
        validation_mae
    ) = evaluate(
        validation_loader
    )


    print(
        f"Epoch {epoch:02d}/{EPOCHS} | "
        f"Train MSE: {train_mse:.4f} | "
        f"Val MSE: {validation_mse:.4f} | "
        f"Val RMSE: {validation_rmse:.4f} | "
        f"Val MAE: {validation_mae:.4f}"
    )


    history.append(
        {
            "epoch": epoch,
            "train_mse": train_mse,
            "validation_mse":
                validation_mse,
            "validation_rmse":
                validation_rmse,
            "validation_mae":
                validation_mae,
        }
    )


    if (
        validation_mse
        < best_validation_mse
    ):

        best_validation_mse = (
            validation_mse
        )

        best_state = copy.deepcopy(
            model.state_dict()
        )


# ============================================================
# 18. RESTAURATION DU MEILLEUR MODELE
# ============================================================

if best_state is None:

    raise RuntimeError(
        "Aucun modèle valide obtenu."
    )


model.load_state_dict(
    best_state
)


# ============================================================
# 19. SAUVEGARDE
# ============================================================

OUTPUT_MODEL_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

HISTORY_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)


torch.save(
    {
        "model_state_dict":
            model.state_dict(),

        "architecture":
            "resnet50",

        "pretraining":
            "ImageNet",

        "adaptation_task":
            "VIIRS regression",

        "target_column":
            TARGET_COLUMN,

        "target_transform":
            "log1p",

        "finetuned_block":
            "layer4",

        "feature_dimension":
            2048,

        "random_state":
            RANDOM_STATE,
    },
    OUTPUT_MODEL_FILE
)


history_df = pd.DataFrame(
    history
)


history_df.to_csv(
    HISTORY_FILE,
    index=False
)


# ============================================================
# 20. BILAN
# ============================================================

best_index = (
    history_df[
        "validation_mse"
    ]
    .idxmin()
)


best_row = history_df.loc[
    best_index
]


print()
print("=" * 72)
print("BILAN FINE-TUNING")
print("=" * 72)

print(
    f"Meilleure epoch : "
    f"{int(best_row['epoch'])}"
)

print(
    f"Validation MSE : "
    f"{best_row['validation_mse']:.4f}"
)

print(
    f"Validation RMSE : "
    f"{best_row['validation_rmse']:.4f}"
)

print(
    f"Validation MAE : "
    f"{best_row['validation_mae']:.4f}"
)

print()
print(
    f"Modèle sauvegardé : "
    f"{OUTPUT_MODEL_FILE}"
)

print(
    f"Historique sauvegardé : "
    f"{HISTORY_FILE}"
)

print()
print(
    "FINE-TUNING VIIRS TERMINE AVEC SUCCES"
)

print("=" * 72)