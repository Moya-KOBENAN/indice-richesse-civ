"""Étape historique isolée : exécution explicite uniquement via main()."""


def main():
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
    # 06B - ADAPTATION DE RESNET-50 AUX LUMIERES NOCTURNES
    # Projet : Indice de richesse - Côte d'Ivoire
    #
    # Objectif :
    # Sentinel-2 -> ResNet-50 ImageNet -> log1p(VIIRS)
    #
    # Cette étape adapte le CNN au domaine satellitaire avant
    # l'extraction des nouvelles deep features.
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

    MODEL_DIR = (
        DHS_MODELS_DIR
    )

    TABLE_DIR = (
        DHS_TABLES_DIR
    )

    MODEL_FILE = (
        MODEL_DIR
        / "resnet50_viirs_adapted.pt"
    )

    HISTORY_FILE = (
        TABLE_DIR
        / "resnet50_viirs_training_history.csv"
    )


    EXPECTED_CLUSTERS = 539

    TARGET_COLUMN = "viirs_2u5r_mean"

    BATCH_SIZE = 16

    EPOCHS_HEAD = 15

    LEARNING_RATE = 1e-3

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


    print("=" * 70)
    print("ADAPTATION RESNET-50 AUX LUMIERES NOCTURNES")
    print("=" * 70)

    print(f"Device utilisé : {DEVICE}")


    # ============================================================
    # 4. IDENTIFICATION DES CLUSTERS
    # ============================================================

    def extract_cluster_id(path):

        match = re.search(
            r"cluster_(\d+)",
            path.stem
        )

        if match is None:

            raise ValueError(
                f"Cluster impossible à identifier : {path.name}"
            )

        return int(
            match.group(1)
        )


    # ============================================================
    # 5. PRETRAITEMENT
    # ============================================================

    weights = ResNet50_Weights.DEFAULT

    preprocess = weights.transforms()


    def load_satellite_image(path):

        with rasterio.open(path) as src:

            image = src.read().astype(
                np.float32
            )

        # rasterio :
        # bandes x hauteur x largeur
        #
        # PIL :
        # hauteur x largeur x bandes

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

        # EXACTEMENT la même transformation
        # que dans 06_extract_cnn_features.py

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

        tensor = preprocess(
            pil_image
        )

        return tensor


    # ============================================================
    # 6. CHARGEMENT DES DONNEES SOCIO-SPATIALES
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
            "La cible VIIRS contient des valeurs manquantes."
        )


    if (df[TARGET_COLUMN] < 0).any():

        raise ValueError(
            "La cible VIIRS contient des valeurs négatives. "
            "log1p ne peut pas être appliqué comme prévu."
        )


    # Transformation utilisée pour réduire
    # l'asymétrie de VIIRS

    df["viirs_target_log"] = np.log1p(
        df[TARGET_COLUMN]
    )


    print(
        f"Nombre de clusters dans la base : {len(df)}"
    )

    print(
        f"Cible : log1p({TARGET_COLUMN})"
    )

    print(
        "Valeurs VIIRS nulles :",
        int(
            (df[TARGET_COLUMN] == 0).sum()
        )
    )

    print(
        "Valeurs manquantes :",
        int(
            df[TARGET_COLUMN].isna().sum()
        )
    )


    # ============================================================
    # 7. ASSOCIATION CLUSTER <-> IMAGE
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


    if len(image_map) != EXPECTED_CLUSTERS:

        raise ValueError(
            "Des identifiants d'images sont dupliqués."
        )


    df["image_path"] = df["cluster"].map(
        image_map
    )


    if df["image_path"].isna().any():

        missing = (
            df.loc[
                df["image_path"].isna(),
                "cluster"
            ]
            .tolist()
        )

        raise ValueError(
            f"Images manquantes pour clusters : {missing[:10]}"
        )


    print(
        f"Images Sentinel-2 associées : "
        f"{df['image_path'].notna().sum()}"
    )


    # ============================================================
    # 8. DATASET PYTORCH
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
    # 9. SEPARATION TRAIN / VALIDATION
    # ============================================================

    n_total = len(dataset)

    n_validation = int(
        round(
            n_total
            * VALIDATION_RATIO
        )
    )

    n_train = (
        n_total
        - n_validation
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
        f"Train : {len(train_dataset)} clusters"
    )

    print(
        f"Validation : {len(validation_dataset)} clusters"
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
    # 10. RESNET-50
    # ============================================================

    model = resnet50(
        weights=weights
    )


    # ============================================================
    # 11. GEL DU BACKBONE
    # ============================================================

    for parameter in model.parameters():

        parameter.requires_grad = False


    # ============================================================
    # 12. NOUVELLE TETE DE REGRESSION VIIRS
    # ============================================================

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


    model = model.to(
        DEVICE
    )


    # ============================================================
    # 13. LOSS + OPTIMIZER
    # ============================================================

    criterion = nn.MSELoss()


    optimizer = torch.optim.Adam(
        model.fc.parameters(),
        lr=LEARNING_RATE
    )


    # ============================================================
    # 14. FONCTION D'EVALUATION
    # ============================================================

    def evaluate(loader):

        model.eval()

        losses = []

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


                output = (
                    model(images)
                    .squeeze(1)
                )


                loss = criterion(
                    output,
                    target
                )


                losses.append(
                    loss.item()
                )


                predictions.extend(
                    output
                    .detach()
                    .cpu()
                    .numpy()
                )


                targets.extend(
                    target
                    .detach()
                    .cpu()
                    .numpy()
                )


        predictions = np.array(
            predictions
        )

        targets = np.array(
            targets
        )


        mse = np.mean(
            (
                predictions
                - targets
            ) ** 2
        )

        rmse = np.sqrt(
            mse
        )


        mae = np.mean(
            np.abs(
                predictions
                - targets
            )
        )


        return (
            float(np.mean(losses)),
            float(rmse),
            float(mae)
        )


    # ============================================================
    # 15. ENTRAINEMENT DE LA TETE
    # ============================================================

    print()
    print("=" * 70)
    print("PHASE 1 - ENTRAINEMENT DE LA TETE VIIRS")
    print("=" * 70)


    history = []

    best_validation_loss = float(
        "inf"
    )

    best_model_state = None


    for epoch in range(
        1,
        EPOCHS_HEAD + 1
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


            output = (
                model(images)
                .squeeze(1)
            )


            loss = criterion(
                output,
                target
            )


            loss.backward()

            optimizer.step()


            train_losses.append(
                loss.item()
            )


        train_loss = float(
            np.mean(
                train_losses
            )
        )


        validation_loss, validation_rmse, validation_mae = (
            evaluate(
                validation_loader
            )
        )


        print(
            f"Epoch {epoch:02d}/{EPOCHS_HEAD} | "
            f"Train MSE: {train_loss:.4f} | "
            f"Val MSE: {validation_loss:.4f} | "
            f"Val RMSE: {validation_rmse:.4f} | "
            f"Val MAE: {validation_mae:.4f}"
        )


        history.append(
            {
                "epoch": epoch,
                "train_mse": train_loss,
                "validation_mse": validation_loss,
                "validation_rmse": validation_rmse,
                "validation_mae": validation_mae,
            }
        )


        if validation_loss < best_validation_loss:

            best_validation_loss = (
                validation_loss
            )

            best_model_state = copy.deepcopy(
                model.state_dict()
            )


    # ============================================================
    # 16. RESTAURATION DU MEILLEUR MODELE
    # ============================================================

    if best_model_state is None:

        raise RuntimeError(
            "Aucun modèle valide n'a été obtenu."
        )


    model.load_state_dict(
        best_model_state
    )


    # ============================================================
    # 17. SAUVEGARDE
    # ============================================================

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    TABLE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    torch.save(
        {
            "model_state_dict":
                model.state_dict(),

            "target_column":
                TARGET_COLUMN,

            "target_transform":
                "log1p",

            "architecture":
                "resnet50",

            "pretraining":
                "ImageNet",

            "adaptation_task":
                "VIIRS regression",

            "feature_dimension":
                2048,

            "random_state":
                RANDOM_STATE,

            "validation_ratio":
                VALIDATION_RATIO,
        },
        MODEL_FILE
    )


    history_df = pd.DataFrame(
        history
    )


    history_df.to_csv(
        HISTORY_FILE,
        index=False
    )


    # ============================================================
    # 18. BILAN
    # ============================================================

    best_epoch = int(
        history_df.loc[
            history_df[
                "validation_mse"
            ].idxmin(),
            "epoch"
        ]
    )


    best_row = history_df.loc[
        history_df[
            "validation_mse"
        ].idxmin()
    ]


    print()
    print("=" * 70)
    print("BILAN ADAPTATION VIIRS")
    print("=" * 70)

    print(
        f"Meilleure epoch : {best_epoch}"
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
        f"Modèle sauvegardé : {MODEL_FILE}"
    )

    print(
        f"Historique sauvegardé : {HISTORY_FILE}"
    )

    print()
    print(
        "PHASE 1 TERMINEE AVEC SUCCES"
    )

    print("=" * 70)

if __name__ == "__main__":
    main()
