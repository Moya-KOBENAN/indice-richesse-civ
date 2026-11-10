"""Étape historique isolée : exécution explicite uniquement via main()."""


def main():
    """
ETAPE 12
Adaptation nationale de ResNet-50 au domaine satellitaire ivoirien
par classification des niveaux de luminosite nocturne VIIRS.

Entrees
-------
- data/raw/satellite_national/grid_XXXX_sentinel2_2021.tif
- data/processed/common/cote_ivoire_national_grid_viirs_diagnostic.csv

Cible
-----
viirs_class_5 : 5 classes de luminosite nocturne.

Strategie
---------
1. ResNet-50 pre-entraine ImageNet.
2. Entrainement d'une nouvelle tete de classification.
3. Fine-tuning de layer4 + tete.
4. Separation spatiale train/validation.
5. Evaluation : accuracy, balanced accuracy, macro-F1.
"""


    # Résolution des chemins indépendante du répertoire de lancement.
    from pathlib import Path
    import sys

    _PROJECT_ROOT = Path(__file__).resolve().parents[3]
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))

    from src.utils.paths import (
        COMMON_MODELS_DIR,
        COMMON_PROCESSED_DIR,
        COMMON_TABLES_DIR,
        NATIONAL_SATELLITE_DIR,
        PROJECT_ROOT,
        ensure_layout,
    )

    ensure_layout('common')


    import random
    import json

    import numpy as np
    import pandas as pd
    import rasterio

    from PIL import Image

    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader

    from torchvision.models import resnet50, ResNet50_Weights

    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        f1_score,
        confusion_matrix,
    )
    from sklearn.utils.class_weight import compute_class_weight

    from pyproj import Transformer


    # ============================================================
    # CONFIGURATION
    # ============================================================

    ROOT = PROJECT_ROOT

    CSV_PATH = (
        COMMON_PROCESSED_DIR / 'cote_ivoire_national_grid_viirs_diagnostic.csv'
    )

    IMAGE_DIR = NATIONAL_SATELLITE_DIR

    MODEL_DIR = COMMON_MODELS_DIR
    TABLE_DIR = COMMON_TABLES_DIR

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)

    HEAD_MODEL_PATH = MODEL_DIR / "resnet50_national_viirs_head.pt"
    FINAL_MODEL_PATH = MODEL_DIR / "resnet50_national_viirs_finetuned.pt"

    HISTORY_PATH = TABLE_DIR / "national_viirs_classifier_history.csv"
    SPLIT_PATH = TABLE_DIR / "national_viirs_spatial_split.csv"
    METRICS_PATH = TABLE_DIR / "national_viirs_classifier_metrics.csv"
    CONFUSION_PATH = TABLE_DIR / "national_viirs_confusion_matrix.csv"

    SEED = 42
    NUM_CLASSES = 5

    BATCH_SIZE = 16

    HEAD_EPOCHS = 10
    FINETUNE_EPOCHS = 10

    HEAD_LR = 1e-3
    LAYER4_LR = 1e-5
    FINETUNE_HEAD_LR = 1e-4

    BLOCK_SIZE_M = 100_000

    NUM_WORKERS = 0


    # ============================================================
    # REPRODUCTIBILITE
    # ============================================================

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if torch.backends.mps.is_available():
        DEVICE = torch.device("mps")
    elif torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    else:
        DEVICE = torch.device("cpu")

    print("=" * 72)
    print("ETAPE 12 - ADAPTATION NATIONALE RESNET-50 -> VIIRS")
    print("=" * 72)
    print("Device :", DEVICE)


    # ============================================================
    # CHARGEMENT DES DONNEES
    # ============================================================

    df = pd.read_csv(CSV_PATH)

    required_cols = {
        "grid_id",
        "longitude",
        "latitude",
        "viirs_class_5",
    }

    missing_cols = required_cols - set(df.columns)

    if missing_cols:
        raise ValueError(
            f"Colonnes manquantes dans le CSV : {missing_cols}"
        )

    if df["grid_id"].duplicated().any():
        raise ValueError("grid_id contient des doublons.")

    if df[list(required_cols)].isna().any().any():
        raise ValueError("Valeurs manquantes detectees.")

    df["grid_id"] = df["grid_id"].astype(int)
    df["viirs_class_5"] = df["viirs_class_5"].astype(int)


    # ============================================================
    # ASSOCIATION AUX IMAGES
    # ============================================================

    def find_image(grid_id):
        filename = f"grid_{grid_id:04d}_sentinel2_2021.tif"
        return IMAGE_DIR / filename


    df["image_path"] = df["grid_id"].apply(find_image)

    df["image_exists"] = df["image_path"].apply(lambda p: p.exists())

    missing_images = df.loc[~df["image_exists"]]

    print()
    print("Nombre de points :", len(df))
    print("Images trouvees :", int(df["image_exists"].sum()))
    print("Images manquantes :", len(missing_images))

    if len(missing_images) > 0:
        print(missing_images[["grid_id", "image_path"]])
        raise FileNotFoundError(
            "Certaines images Sentinel-2 sont manquantes."
        )


    # ============================================================
    # DISTRIBUTION DES CLASSES
    # ============================================================

    print()
    print("Distribution VIIRS :")

    distribution = (
        df["viirs_class_5"]
        .value_counts()
        .sort_index()
    )

    for classe, n in distribution.items():
        pct = 100 * n / len(df)
        print(
            f"Classe {classe} : {n:4d} "
            f"({pct:5.2f} %)"
        )


    # ============================================================
    # CREATION DES BLOCS SPATIAUX
    # ============================================================

    transformer = Transformer.from_crs(
        "EPSG:4326",
        "EPSG:32630",
        always_xy=True,
    )

    x, y = transformer.transform(
        df["longitude"].to_numpy(),
        df["latitude"].to_numpy(),
    )

    df["utm_x"] = x
    df["utm_y"] = y

    df["block_x"] = np.floor(
        df["utm_x"] / BLOCK_SIZE_M
    ).astype(int)

    df["block_y"] = np.floor(
        df["utm_y"] / BLOCK_SIZE_M
    ).astype(int)

    df["spatial_block"] = (
        df["block_x"].astype(str)
        + "_"
        + df["block_y"].astype(str)
    )

    print()
    print(
        "Nombre de blocs spatiaux :",
        df["spatial_block"].nunique(),
    )


    # ============================================================
    # SPLIT SPATIAL
    # ============================================================

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=SEED,
    )

    train_idx, val_idx = next(
        splitter.split(
            df,
            groups=df["spatial_block"],
        )
    )

    df["split"] = "train"
    df.loc[val_idx, "split"] = "validation"

    train_df = df.iloc[train_idx].copy()
    val_df = df.iloc[val_idx].copy()

    print()
    print("TRAIN :", len(train_df))
    print("VALIDATION :", len(val_df))

    train_blocks = set(train_df["spatial_block"])
    val_blocks = set(val_df["spatial_block"])

    overlap = train_blocks.intersection(val_blocks)

    print("Chevauchement blocs :", len(overlap))

    if overlap:
        raise RuntimeError(
            "Des blocs spatiaux apparaissent dans train et validation."
        )

    print()
    print("Classes TRAIN :")
    print(
        train_df["viirs_class_5"]
        .value_counts()
        .sort_index()
    )

    print()
    print("Classes VALIDATION :")
    print(
        val_df["viirs_class_5"]
        .value_counts()
        .sort_index()
    )

    if train_df["viirs_class_5"].nunique() != NUM_CLASSES:
        raise RuntimeError(
            "Toutes les classes ne sont pas presentes dans TRAIN."
        )

    if val_df["viirs_class_5"].nunique() != NUM_CLASSES:
        raise RuntimeError(
            "Toutes les classes ne sont pas presentes dans VALIDATION."
        )

    df[
        [
            "grid_id",
            "longitude",
            "latitude",
            "viirs_class_5",
            "spatial_block",
            "split",
        ]
    ].to_csv(
        SPLIT_PATH,
        index=False,
    )


    # ============================================================
    # PREPROCESSING IMAGENET
    # ============================================================

    weights = ResNet50_Weights.DEFAULT
    transform = weights.transforms()


    class SatelliteDataset(Dataset):

        def __init__(self, dataframe):
            self.df = dataframe.reset_index(drop=True)

        def __len__(self):
            return len(self.df)

        def __getitem__(self, idx):

            row = self.df.iloc[idx]

            path = Path(row["image_path"])

            with rasterio.open(path) as src:
                arr = src.read().astype(np.float32)

            if arr.shape[0] != 3:
                raise ValueError(
                    f"{path.name} : {arr.shape[0]} bandes au lieu de 3."
                )

            arr = np.nan_to_num(
                arr,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )

            # Meme logique que l'extraction ImageNet precedente :
            # reflectance Sentinel-2 ramenee visuellement vers RGB.
            arr = np.clip(arr, 0.0, 0.30)
            arr = arr / 0.30
            arr = arr * 255.0

            arr = arr.astype(np.uint8)

            arr = np.transpose(
                arr,
                (1, 2, 0),
            )

            image = Image.fromarray(
                arr,
                mode="RGB",
            )

            image = transform(image)

            target = int(
                row["viirs_class_5"]
            )

            grid_id = int(
                row["grid_id"]
            )

            return image, target, grid_id


    train_dataset = SatelliteDataset(train_df)
    val_dataset = SatelliteDataset(val_df)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )


    # ============================================================
    # POIDS DES CLASSES
    # ============================================================

    classes = np.arange(NUM_CLASSES)

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=train_df["viirs_class_5"].to_numpy(),
    )

    class_weights_tensor = torch.tensor(
        class_weights,
        dtype=torch.float32,
        device=DEVICE,
    )

    print()
    print("Poids des classes :")

    for i, w in enumerate(class_weights):
        print(
            f"Classe {i} : {w:.4f}"
        )


    # ============================================================
    # MODELE
    # ============================================================

    model = resnet50(
        weights=weights
    )

    in_features = model.fc.in_features

    model.fc = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.30),
        nn.Linear(256, NUM_CLASSES),
    )

    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights_tensor
    )


    # ============================================================
    # FONCTIONS TRAIN / EVALUATION
    # ============================================================

    def train_one_epoch(model, loader, optimizer):

        model.train()

        total_loss = 0.0
        y_true = []
        y_pred = []

        for images, targets, _ in loader:

            images = images.to(DEVICE)
            targets = targets.to(DEVICE)

            optimizer.zero_grad()

            logits = model(images)

            loss = criterion(
                logits,
                targets,
            )

            loss.backward()
            optimizer.step()

            total_loss += (
                loss.item()
                * images.size(0)
            )

            preds = torch.argmax(
                logits,
                dim=1,
            )

            y_true.extend(
                targets.detach().cpu().numpy()
            )

            y_pred.extend(
                preds.detach().cpu().numpy()
            )

        loss_mean = (
            total_loss
            / len(loader.dataset)
        )

        bal_acc = balanced_accuracy_score(
            y_true,
            y_pred,
        )

        macro_f1 = f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        )

        return loss_mean, bal_acc, macro_f1


    @torch.no_grad()
    def evaluate(model, loader):

        model.eval()

        total_loss = 0.0

        y_true = []
        y_pred = []
        grid_ids = []

        for images, targets, ids in loader:

            images = images.to(DEVICE)
            targets = targets.to(DEVICE)

            logits = model(images)

            loss = criterion(
                logits,
                targets,
            )

            total_loss += (
                loss.item()
                * images.size(0)
            )

            preds = torch.argmax(
                logits,
                dim=1,
            )

            y_true.extend(
                targets.cpu().numpy()
            )

            y_pred.extend(
                preds.cpu().numpy()
            )

            grid_ids.extend(
                ids.numpy()
            )

        loss_mean = (
            total_loss
            / len(loader.dataset)
        )

        accuracy = accuracy_score(
            y_true,
            y_pred,
        )

        bal_acc = balanced_accuracy_score(
            y_true,
            y_pred,
        )

        macro_f1 = f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        )

        return {
            "loss": loss_mean,
            "accuracy": accuracy,
            "balanced_accuracy": bal_acc,
            "macro_f1": macro_f1,
            "y_true": np.array(y_true),
            "y_pred": np.array(y_pred),
            "grid_ids": np.array(grid_ids),
        }


    # ============================================================
    # PHASE 1 : TETE SEULE
    # ============================================================

    print()
    print("=" * 72)
    print("PHASE 1 - ENTRAINEMENT DE LA TETE")
    print("=" * 72)

    for param in model.parameters():
        param.requires_grad = False

    for param in model.fc.parameters():
        param.requires_grad = True

    optimizer = torch.optim.Adam(
        model.fc.parameters(),
        lr=HEAD_LR,
    )

    history = []

    best_score = -np.inf

    for epoch in range(
        1,
        HEAD_EPOCHS + 1,
    ):

        train_loss, train_bal, train_f1 = train_one_epoch(
            model,
            train_loader,
            optimizer,
        )

        val_metrics = evaluate(
            model,
            val_loader,
        )

        print(
            f"[HEAD {epoch:02d}/{HEAD_EPOCHS}] "
            f"Train loss={train_loss:.4f} | "
            f"Train BalAcc={train_bal:.4f} | "
            f"Val loss={val_metrics['loss']:.4f} | "
            f"Val BalAcc={val_metrics['balanced_accuracy']:.4f} | "
            f"Val MacroF1={val_metrics['macro_f1']:.4f}"
        )

        history.append(
            {
                "phase": "head",
                "epoch": epoch,
                "train_loss": train_loss,
                "train_balanced_accuracy": train_bal,
                "train_macro_f1": train_f1,
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_balanced_accuracy": val_metrics["balanced_accuracy"],
                "val_macro_f1": val_metrics["macro_f1"],
            }
        )

        score = val_metrics[
            "balanced_accuracy"
        ]

        if score > best_score:

            best_score = score

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "phase": "head",
                    "best_balanced_accuracy": best_score,
                    "num_classes": NUM_CLASSES,
                    "seed": SEED,
                },
                HEAD_MODEL_PATH,
            )


    # ============================================================
    # RECHARGEMENT DU MEILLEUR HEAD
    # ============================================================

    checkpoint = torch.load(
        HEAD_MODEL_PATH,
        map_location=DEVICE,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )


    # ============================================================
    # PHASE 2 : FINE-TUNING LAYER4 + TETE
    # ============================================================

    print()
    print("=" * 72)
    print("PHASE 2 - FINE-TUNING LAYER4 + TETE")
    print("=" * 72)

    for param in model.parameters():
        param.requires_grad = False

    for param in model.layer4.parameters():
        param.requires_grad = True

    for param in model.fc.parameters():
        param.requires_grad = True

    optimizer = torch.optim.Adam(
        [
            {
                "params": model.layer4.parameters(),
                "lr": LAYER4_LR,
            },
            {
                "params": model.fc.parameters(),
                "lr": FINETUNE_HEAD_LR,
            },
        ]
    )

    best_score = -np.inf

    for epoch in range(
        1,
        FINETUNE_EPOCHS + 1,
    ):

        train_loss, train_bal, train_f1 = train_one_epoch(
            model,
            train_loader,
            optimizer,
        )

        val_metrics = evaluate(
            model,
            val_loader,
        )

        print(
            f"[FT {epoch:02d}/{FINETUNE_EPOCHS}] "
            f"Train loss={train_loss:.4f} | "
            f"Train BalAcc={train_bal:.4f} | "
            f"Val loss={val_metrics['loss']:.4f} | "
            f"Val BalAcc={val_metrics['balanced_accuracy']:.4f} | "
            f"Val MacroF1={val_metrics['macro_f1']:.4f}"
        )

        history.append(
            {
                "phase": "finetune",
                "epoch": epoch,
                "train_loss": train_loss,
                "train_balanced_accuracy": train_bal,
                "train_macro_f1": train_f1,
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
                "val_balanced_accuracy": val_metrics["balanced_accuracy"],
                "val_macro_f1": val_metrics["macro_f1"],
            }
        )

        score = val_metrics[
            "balanced_accuracy"
        ]

        if score > best_score:

            best_score = score

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "phase": "finetune",
                    "best_balanced_accuracy": best_score,
                    "num_classes": NUM_CLASSES,
                    "seed": SEED,
                    "block_size_m": BLOCK_SIZE_M,
                },
                FINAL_MODEL_PATH,
            )


    # ============================================================
    # HISTORIQUE
    # ============================================================

    history_df = pd.DataFrame(
        history
    )

    history_df.to_csv(
        HISTORY_PATH,
        index=False,
    )


    # ============================================================
    # EVALUATION FINALE
    # ============================================================

    checkpoint = torch.load(
        FINAL_MODEL_PATH,
        map_location=DEVICE,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    final_metrics = evaluate(
        model,
        val_loader,
    )

    metrics_df = pd.DataFrame(
        [
            {
                "n_total": len(df),
                "n_train": len(train_df),
                "n_validation": len(val_df),
                "n_spatial_blocks": df[
                    "spatial_block"
                ].nunique(),
                "block_size_km": (
                    BLOCK_SIZE_M / 1000
                ),
                "accuracy": final_metrics[
                    "accuracy"
                ],
                "balanced_accuracy": final_metrics[
                    "balanced_accuracy"
                ],
                "macro_f1": final_metrics[
                    "macro_f1"
                ],
                "validation_loss": final_metrics[
                    "loss"
                ],
            }
        ]
    )

    metrics_df.to_csv(
        METRICS_PATH,
        index=False,
    )

    cm = confusion_matrix(
        final_metrics["y_true"],
        final_metrics["y_pred"],
        labels=np.arange(NUM_CLASSES),
    )

    cm_df = pd.DataFrame(
        cm,
        index=[
            f"true_{i}"
            for i in range(NUM_CLASSES)
        ],
        columns=[
            f"pred_{i}"
            for i in range(NUM_CLASSES)
        ],
    )

    cm_df.to_csv(
        CONFUSION_PATH
    )


    # ============================================================
    # BILAN
    # ============================================================

    print()
    print("=" * 72)
    print("BILAN FINAL")
    print("=" * 72)

    print(
        "Accuracy :",
        f"{final_metrics['accuracy']:.4f}",
    )

    print(
        "Balanced accuracy :",
        f"{final_metrics['balanced_accuracy']:.4f}",
    )

    print(
        "Macro-F1 :",
        f"{final_metrics['macro_f1']:.4f}",
    )

    print()
    print("Matrice de confusion :")
    print(cm_df)

    print()
    print("Modele sauvegarde :")
    print(FINAL_MODEL_PATH)

    print()
    print("Historique :")
    print(HISTORY_PATH)

    print()
    print("Split spatial :")
    print(SPLIT_PATH)

    print()
    print("Metriques :")
    print(METRICS_PATH)

    print()
    print("ETAPE 12 TERMINEE")
    print("=" * 72)


if __name__ == "__main__":
    main()
