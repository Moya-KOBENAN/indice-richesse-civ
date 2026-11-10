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
    # 06 - EXTRACTION DES CARACTERISTIQUES CNN
    # Projet : Indice de richesse - Côte d'Ivoire
    # Modèle : ResNet-50 pré-entraîné sur ImageNet
    # ============================================================


    # ============================================================
    # 1. CHEMINS ET PARAMETRES
    # ============================================================


    SATELLITE_DIR = (
        DHS_SATELLITE_DIR
    )

    OUTPUT_FILE = (
        DHS_PROCESSED_DIR / 'cnn_features_resnet50.csv'
    )

    EXPECTED_CLUSTERS = 539

    BATCH_SIZE = 16


    # ============================================================
    # 2. CHOIX DU DEVICE
    # ============================================================

    if torch.backends.mps.is_available():
        DEVICE = torch.device("mps")

    elif torch.cuda.is_available():
        DEVICE = torch.device("cuda")

    else:
        DEVICE = torch.device("cpu")


    print("=" * 60)
    print("EXTRACTION DES FEATURES CNN")
    print("=" * 60)

    print(f"Device utilisé : {DEVICE}")


    # ============================================================
    # 3. CHARGEMENT DE RESNET-50
    # ============================================================

    weights = ResNet50_Weights.DEFAULT

    model = resnet50(
        weights=weights
    )

    # On retire la dernière couche de classification.
    # ResNet-50 retourne alors 2048 caractéristiques
    # pour chaque image.

    model.fc = nn.Identity()

    model = model.to(DEVICE)

    model.eval()

    preprocess = weights.transforms()


    print("Modèle : ResNet-50 pré-entraîné ImageNet")
    print("Nombre de features attendu : 2048")


    # ============================================================
    # 4. IDENTIFICATION DU CLUSTER
    # ============================================================

    def extract_cluster_id(path):
        """
    Extrait l'identifiant DHS du cluster.

    Exemple :
    cluster_001_sentinel2_2021.tif
    -> cluster = 1
    """

        match = re.search(
            r"cluster_(\d+)",
            path.stem
        )

        if match is None:
            raise ValueError(
                f"Impossible d'identifier le cluster "
                f"dans le fichier : {path.name}"
            )

        return int(match.group(1))


    # ============================================================
    # 5. LECTURE ET PRETRAITEMENT D'UNE IMAGE
    # ============================================================

    def load_satellite_image(path):
        """
    Lit une image Sentinel-2 RGB et applique
    le prétraitement nécessaire pour ResNet-50.
    """

        with rasterio.open(path) as src:

            image = src.read().astype(
                np.float32
            )

        # Format rasterio :
        # (bandes, hauteur, largeur)
        #
        # Format image :
        # (hauteur, largeur, bandes)

        image = np.transpose(
            image,
            (1, 2, 0)
        )

        # Nettoyage éventuel des NaN et infinis

        image = np.nan_to_num(
            image,
            nan=0.0,
            posinf=0.0,
            neginf=0.0
        )

        # Les valeurs Sentinel-2 sont des réflectances.
        # On applique une plage fixe 0 - 0.30.
        #
        # Important :
        # on utilise la même transformation pour toutes
        # les images afin de conserver les différences
        # de luminosité entre clusters.

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

        # Conversion en image RGB

        pil_image = Image.fromarray(
            image,
            mode="RGB"
        )

        # Prétraitement ImageNet de ResNet-50

        tensor = preprocess(
            pil_image
        )

        return tensor


    # ============================================================
    # 6. SAUVEGARDE
    # ============================================================

    def save_features(rows):
        """
    Sauvegarde les caractéristiques CNN.
    """

        df = pd.DataFrame(
            rows
        )

        OUTPUT_FILE.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        df.to_csv(
            OUTPUT_FILE,
            index=False
        )


    # ============================================================
    # 7. RECUPERATION DES IMAGES
    # ============================================================

    image_paths = list(
        SATELLITE_DIR.glob("*.tif")
    )

    # Tri par numéro réel du cluster :
    # 1, 2, 3, ..., 539

    image_paths = sorted(
        image_paths,
        key=extract_cluster_id
    )


    print(
        f"Nombre d'images trouvées : "
        f"{len(image_paths)}"
    )


    # ============================================================
    # 8. CONTROLES AVANT CNN
    # ============================================================

    if len(image_paths) != EXPECTED_CLUSTERS:

        raise ValueError(
            f"{EXPECTED_CLUSTERS} images attendues, "
            f"{len(image_paths)} trouvées."
        )


    cluster_ids = [
        extract_cluster_id(path)
        for path in image_paths
    ]


    unique_cluster_ids = set(
        cluster_ids
    )


    print(
        f"Nombre d'identifiants uniques : "
        f"{len(unique_cluster_ids)}"
    )

    print(
        f"Premiers clusters : "
        f"{cluster_ids[:5]}"
    )

    print(
        f"Derniers clusters : "
        f"{cluster_ids[-5:]}"
    )


    if len(unique_cluster_ids) != EXPECTED_CLUSTERS:

        duplicates = (
            pd.Series(cluster_ids)
            .loc[
                pd.Series(cluster_ids)
                .duplicated()
            ]
            .tolist()
        )

        raise ValueError(
            "Des identifiants de clusters sont dupliqués "
            f"avant même l'extraction CNN : {duplicates[:10]}"
        )


    # ============================================================
    # 9. EXTRACTION DES FEATURES CNN
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

            batch_cluster_ids = []


            # --------------------------------------------
            # Préparation du batch
            # --------------------------------------------

            for path in batch_paths:

                cluster_id = (
                    extract_cluster_id(path)
                )

                tensor = (
                    load_satellite_image(path)
                )

                tensors.append(
                    tensor
                )

                batch_cluster_ids.append(
                    cluster_id
                )


            # --------------------------------------------
            # Création du batch PyTorch
            # --------------------------------------------

            batch = torch.stack(
                tensors
            )

            batch = batch.to(
                DEVICE
            )


            # --------------------------------------------
            # Passage dans ResNet-50
            # --------------------------------------------

            features = model(
                batch
            )


            # Retour CPU / NumPy

            features = (
                features
                .cpu()
                .numpy()
            )


            # --------------------------------------------
            # Stockage
            # --------------------------------------------

            for (
                cluster_id,
                feature_vector
            ) in zip(
                batch_cluster_ids,
                features
            ):

                row = {
                    "cluster": cluster_id
                }


                for j, value in enumerate(
                    feature_vector
                ):

                    row[
                        f"cnn_{j:04d}"
                    ] = float(
                        value
                    )


                rows.append(
                    row
                )


            # --------------------------------------------
            # Progression
            # --------------------------------------------

            current = min(
                start + BATCH_SIZE,
                len(image_paths)
            )

            print(
                f"Clusters traités : "
                f"{current}/"
                f"{len(image_paths)}"
            )


            # Sauvegarde progressive

            save_features(
                rows
            )


    # ============================================================
    # 10. CREATION DE LA BASE FINALE
    # ============================================================

    features_df = pd.DataFrame(
        rows
    )


    # ============================================================
    # 11. CONTROLES FINAUX
    # ============================================================

    print("\n" + "=" * 60)

    print("BILAN CNN")

    print("=" * 60)


    n_clusters = len(
        features_df
    )

    n_features = (
        features_df.shape[1] - 1
    )

    n_missing = (
        features_df
        .isna()
        .sum()
        .sum()
    )

    n_duplicates = (
        features_df[
            "cluster"
        ]
        .duplicated()
        .sum()
    )


    print(
        f"Nombre de clusters : "
        f"{n_clusters}"
    )

    print(
        f"Nombre de features CNN : "
        f"{n_features}"
    )

    print(
        f"Valeurs manquantes : "
        f"{n_missing}"
    )

    print(
        f"Doublons cluster : "
        f"{n_duplicates}"
    )


    # ============================================================
    # 12. VERIFICATIONS DE SECURITE
    # ============================================================

    if n_clusters != EXPECTED_CLUSTERS:

        raise ValueError(
            "Le nombre final de clusters "
            "est incorrect."
        )


    if n_features != 2048:

        raise ValueError(
            f"2048 features attendues, "
            f"{n_features} obtenues."
        )


    if n_duplicates != 0:

        raise ValueError(
            "Des clusters sont présents "
            "plusieurs fois."
        )


    if n_missing != 0:

        raise ValueError(
            "Des valeurs manquantes sont "
            "présentes dans les features CNN."
        )


    # Vérification des valeurs infinies

    feature_columns = [
        column
        for column in features_df.columns
        if column.startswith("cnn_")
    ]

    feature_array = (
        features_df[
            feature_columns
        ]
        .to_numpy()
    )


    if not np.isfinite(
        feature_array
    ).all():

        raise ValueError(
            "Certaines features CNN "
            "contiennent des valeurs infinies."
        )


    # Vérification de la variabilité

    variances = (
        features_df[
            feature_columns
        ]
        .var()
    )

    n_variable_features = (
        variances > 0
    ).sum()


    print(
        f"Features avec variance > 0 : "
        f"{n_variable_features}/2048"
    )


    if n_variable_features == 0:

        raise ValueError(
            "Toutes les features CNN "
            "sont constantes."
        )


    # ============================================================
    # 13. SAUVEGARDE FINALE
    # ============================================================

    features_df = (
        features_df
        .sort_values("cluster")
        .reset_index(drop=True)
    )


    features_df.to_csv(
        OUTPUT_FILE,
        index=False
    )


    print("\nFichier sauvegardé :")

    print(
        OUTPUT_FILE
    )


    print("\n" + "=" * 60)

    print(
        "EXTRACTION CNN TERMINEE AVEC SUCCES"
    )

    print("=" * 60)

if __name__ == "__main__":
    main()
