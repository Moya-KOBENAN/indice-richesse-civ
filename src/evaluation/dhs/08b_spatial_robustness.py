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
        DHS_TABLES_DIR,
        PROJECT_ROOT,
        ensure_layout,
    )

    ensure_layout('dhs')


    import numpy as np
    import pandas as pd

    from sklearn.base import clone
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import Ridge
    from sklearn.metrics import (
        mean_absolute_error,
        mean_squared_error,
        r2_score,
    )
    from sklearn.model_selection import (
        GridSearchCV,
        GroupKFold,
        KFold,
    )
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler


    # ============================================================
    # 08B - ROBUSTESSE DE LA VALIDATION SPATIALE
    # Blocs géographiques réguliers
    #
    # Objectif :
    # compléter la validation géographiquement groupée par KMeans
    # avec une seconde définition, indépendante, des groupes spatiaux.
    #
    # IMPORTANT :
    # cette analyse n'écrase aucun résultat de l'étape 08.
    # ============================================================


    # ============================================================
    # 1. PARAMETRES
    # ============================================================


    MAIN_FILE = (
        DHS_PROCESSED_DIR / 'DHS_CI_2021_richesse_GPS_VIIRS_merged.csv'
    )

    CNN_FILE = (
        DHS_PROCESSED_DIR / 'cnn_features_resnet50.csv'
    )

    OUTPUT_DIR = (
        DHS_TABLES_DIR
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    METRICS_FILE = (
        OUTPUT_DIR
        / "spatial_block_robustness_metrics.csv"
    )

    PREDICTIONS_FILE = (
        OUTPUT_DIR
        / "spatial_block_robustness_predictions.csv"
    )

    BLOCKS_FILE = (
        OUTPUT_DIR
        / "spatial_regular_blocks.csv"
    )

    COMPARISON_FILE = (
        OUTPUT_DIR
        / "validation_methods_comparison.csv"
    )


    TARGET = "wealth_mean"

    VIIRS_VARIABLE = "viirs_2u5r_mean"

    RANDOM_STATE = 42

    # Taille du bloc spatial en kilomètres.
    # 100 km est utilisé comme analyse principale de sensibilité.
    BLOCK_SIZE_KM = 100.0

    N_SPATIAL_FOLDS = 5

    ALPHAS = np.logspace(
        -3,
        4,
        20
    )


    # ============================================================
    # 2. CHARGEMENT
    # ============================================================

    print("=" * 72)
    print("ETAPE 08B - ROBUSTESSE PAR BLOCS SPATIAUX REGULIERS")
    print("=" * 72)

    main_df = pd.read_csv(
        MAIN_FILE
    )

    cnn_df = pd.read_csv(
        CNN_FILE
    )

    print(
        f"Base principale : {main_df.shape}"
    )

    print(
        f"Base CNN : {cnn_df.shape}"
    )


    # ============================================================
    # 3. CONTROLES
    # ============================================================

    required_columns = [
        "cluster",
        TARGET,
        VIIRS_VARIABLE,
        "latitude",
        "longitude",
    ]

    for column in required_columns:

        if column not in main_df.columns:

            raise ValueError(
                f"Variable absente : {column}"
            )


    if main_df["cluster"].duplicated().any():

        raise ValueError(
            "Doublons dans la base principale."
        )


    if cnn_df["cluster"].duplicated().any():

        raise ValueError(
            "Doublons dans la base CNN."
        )


    if main_df[
        ["latitude", "longitude"]
    ].isna().any().any():

        raise ValueError(
            "Coordonnées GPS manquantes."
        )


    # ============================================================
    # 4. FUSION
    # ============================================================

    df = main_df.merge(
        cnn_df,
        on="cluster",
        how="inner",
        validate="one_to_one",
    )


    if len(df) != 539:

        raise ValueError(
            f"539 clusters attendus, "
            f"{len(df)} obtenus."
        )


    # Défragmentation du DataFrame.
    # Cela évite le PerformanceWarning rencontré à l'étape 08.

    df = df.copy()


    print(
        f"Base fusionnée : {df.shape}"
    )


    # ============================================================
    # 5. CONVERSION APPROXIMATIVE EN COORDONNEES KILOMETRIQUES
    # ============================================================

    # Pour construire une grille spatiale, latitude et longitude
    # sont converties approximativement en distances kilométriques.
    #
    # Cette transformation ne sert PAS de variable prédictive.
    # Elle sert uniquement à définir les blocs géographiques.

    mean_latitude_rad = np.radians(
        df["latitude"].mean()
    )


    x_km = (
        df["longitude"].to_numpy(dtype=float)
        * 111.32
        * np.cos(mean_latitude_rad)
    )


    y_km = (
        df["latitude"].to_numpy(dtype=float)
        * 110.57
    )


    # On ramène l'origine de la grille au coin inférieur gauche.

    x_origin = np.min(x_km)

    y_origin = np.min(y_km)


    x_relative = (
        x_km
        - x_origin
    )

    y_relative = (
        y_km
        - y_origin
    )


    # ============================================================
    # 6. CREATION DES BLOCS REGULIERS
    # ============================================================

    block_x = np.floor(
        x_relative
        / BLOCK_SIZE_KM
    ).astype(int)


    block_y = np.floor(
        y_relative
        / BLOCK_SIZE_KM
    ).astype(int)


    # Identifiant textuel unique pour chaque cellule de la grille.

    regular_block = np.array(
        [
            f"{bx}_{by}"
            for bx, by in zip(
                block_x,
                block_y
            )
        ]
    )


    df["regular_block"] = (
        regular_block
    )


    n_blocks = (
        df["regular_block"]
        .nunique()
    )


    print(
        f"\nTaille des blocs : "
        f"{BLOCK_SIZE_KM:.0f} km x "
        f"{BLOCK_SIZE_KM:.0f} km"
    )

    print(
        f"Nombre de blocs occupés : "
        f"{n_blocks}"
    )


    if n_blocks < N_SPATIAL_FOLDS:

        raise ValueError(
            "Nombre de blocs insuffisant pour "
            "la validation croisée spatiale."
        )


    block_counts = (
        df["regular_block"]
        .value_counts()
    )


    print(
        f"Nombre moyen de clusters par bloc : "
        f"{block_counts.mean():.2f}"
    )

    print(
        f"Médiane : "
        f"{block_counts.median():.2f}"
    )

    print(
        f"Minimum : "
        f"{block_counts.min()}"
    )

    print(
        f"Maximum : "
        f"{block_counts.max()}"
    )


    # ============================================================
    # 7. SAUVEGARDE DE LA GRILLE
    # ============================================================

    blocks_table = df[
        [
            "cluster",
            "latitude",
            "longitude",
            "regular_block",
        ]
    ].copy()


    blocks_table[
        "block_size_km"
    ] = BLOCK_SIZE_KM


    blocks_table.to_csv(
        BLOCKS_FILE,
        index=False
    )


    # ============================================================
    # 8. VARIABLE CIBLE
    # ============================================================

    y = df[
        TARGET
    ].to_numpy(
        dtype=float
    )


    if not np.isfinite(y).all():

        raise ValueError(
            "La variable cible contient "
            "des valeurs invalides."
        )


    # ============================================================
    # 9. FEATURES CNN
    # ============================================================

    cnn_columns = [
        column
        for column in df.columns
        if column.startswith("cnn_")
    ]


    if len(cnn_columns) != 2048:

        raise ValueError(
            f"2048 features CNN attendues, "
            f"{len(cnn_columns)} obtenues."
        )


    # ============================================================
    # 10. VIIRS
    # ============================================================

    viirs_raw = df[
        VIIRS_VARIABLE
    ].astype(float)


    if not np.isfinite(
        viirs_raw
    ).all():

        raise ValueError(
            "VIIRS contient des valeurs invalides."
        )


    # Même transformation que dans les étapes 04, 07 et 08.

    viirs_log = np.log1p(
        viirs_raw.clip(
            lower=0
        )
    )


    # ============================================================
    # 11. JEUX DE PREDICTEURS
    # ============================================================

    X_viirs = pd.DataFrame(
        {
            "log_viirs": (
                viirs_log.to_numpy()
            )
        }
    )


    X_cnn = (
        df[
            cnn_columns
        ]
        .copy()
    )


    X_cnn_viirs = pd.concat(
        [
            X_cnn.reset_index(
                drop=True
            ),
            X_viirs.reset_index(
                drop=True
            ),
        ],
        axis=1,
    )


    datasets = {
        "VIIRS": X_viirs,
        "CNN": X_cnn,
        "CNN+VIIRS": X_cnn_viirs,
    }


    print(
        "\nVariables prédictives :"
    )

    print(
        f"VIIRS      : {X_viirs.shape[1]}"
    )

    print(
        f"CNN        : {X_cnn.shape[1]}"
    )

    print(
        f"CNN+VIIRS  : {X_cnn_viirs.shape[1]}"
    )


    # ============================================================
    # 12. PIPELINE RIDGE
    # ============================================================

    base_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "scaler",
                StandardScaler()
            ),
            (
                "ridge",
                Ridge()
            ),
        ]
    )


    # ============================================================
    # 13. VALIDATION SPATIALE PAR BLOCS
    # ============================================================

    groups = df[
        "regular_block"
    ].to_numpy()


    outer_cv = GroupKFold(
        n_splits=N_SPATIAL_FOLDS
    )


    spatial_splits = list(
        outer_cv.split(
            df,
            y,
            groups=groups
        )
    )


    print(
        f"\nNombre de folds spatiaux : "
        f"{len(spatial_splits)}"
    )


    # ============================================================
    # 14. CONTROLE STRICT DES FUITES
    # ============================================================

    fold_assignment = np.full(
        len(df),
        -1,
        dtype=int
    )


    print(
        "\nContrôle des blocs train/test :"
    )


    for fold_number, (
        train_index,
        test_index
    ) in enumerate(
        spatial_splits,
        start=1
    ):

        train_blocks = set(
            groups[
                train_index
            ]
        )

        test_blocks = set(
            groups[
                test_index
            ]
        )


        overlap = (
            train_blocks
            .intersection(
                test_blocks
            )
        )


        if overlap:

            raise ValueError(
                f"Fuite spatiale au fold "
                f"{fold_number}."
            )


        fold_assignment[
            test_index
        ] = fold_number


        print(
            f"Fold {fold_number}: "
            f"train={len(train_index)} "
            f"| test={len(test_index)} "
            f"| blocs train={len(train_blocks)} "
            f"| blocs test={len(test_blocks)} "
            f"| communs={len(overlap)}"
        )


    if np.any(
        fold_assignment == -1
    ):

        raise ValueError(
            "Certains clusters n'ont pas été "
            "attribués à un fold test."
        )


    # ============================================================
    # 15. TABLE DES PREDICTIONS
    # ============================================================

    prediction_table = df[
        [
            "cluster",
            "latitude",
            "longitude",
            "regular_block",
        ]
    ].copy()


    prediction_table[
        "spatial_fold"
    ] = fold_assignment


    prediction_table[
        "wealth_observed"
    ] = y


    # ============================================================
    # 16. MODELISATION
    # ============================================================

    results = []


    for model_name, X in datasets.items():

        print(
            "\n" + "-" * 72
        )

        print(
            f"MODELE : {model_name}"
        )

        print(
            "-" * 72
        )


        predictions = np.full(
            len(df),
            np.nan
        )


        selected_alphas = []


        for fold_number, (
            train_index,
            test_index
        ) in enumerate(
            spatial_splits,
            start=1
        ):

            X_train = X.iloc[
                train_index
            ]

            X_test = X.iloc[
                test_index
            ]

            y_train = y[
                train_index
            ]


            # ----------------------------------------------------
            # Validation interne
            #
            # Le jeu test spatial externe n'est jamais utilisé
            # pour choisir alpha.
            # ----------------------------------------------------

            # Correction : la sélection de alpha respecte aussi les groupes spatiaux.
            # Un KFold aléatoire interne choisirait alpha pour interpoler
            # entre voisins, alors que l'évaluation externe teste l'extrapolation.
            inner_cv = GroupKFold(
                n_splits=5
            )


            search = GridSearchCV(
                estimator=clone(
                    base_pipeline
                ),
                param_grid={
                    "ridge__alpha": ALPHAS
                },
                scoring=(
                    "neg_root_mean_squared_error"
                ),
                cv=inner_cv,

                # Choix volontaire pour limiter
                # l'utilisation de la mémoire.
                n_jobs=1,
            )


            search.fit(
                X_train,
                y_train,
                groups=groups[train_index],
            )


            best_model = (
                search.best_estimator_
            )


            best_alpha = float(
                search.best_params_[
                    "ridge__alpha"
                ]
            )


            selected_alphas.append(
                best_alpha
            )


            predictions[
                test_index
            ] = best_model.predict(
                X_test
            )


            print(
                f"Fold {fold_number:02d}/"
                f"{N_SPATIAL_FOLDS} "
                f"| train={len(train_index)} "
                f"| test={len(test_index)} "
                f"| alpha={best_alpha:.6g}"
            )


        # ========================================================
        # 17. CONTROLES DES PREDICTIONS
        # ========================================================

        if np.isnan(
            predictions
        ).any():

            raise ValueError(
                f"Prédictions manquantes "
                f"pour {model_name}."
            )


        if not np.isfinite(
            predictions
        ).all():

            raise ValueError(
                f"Prédictions non finies "
                f"pour {model_name}."
            )


        # ========================================================
        # 18. METRIQUES
        # ========================================================

        r2 = r2_score(
            y,
            predictions
        )


        rmse = np.sqrt(
            mean_squared_error(
                y,
                predictions
            )
        )


        mae = mean_absolute_error(
            y,
            predictions
        )


        print(
            f"\nR² blocs   : {r2:.4f}"
        )

        print(
            f"RMSE blocs : {rmse:.4f}"
        )

        print(
            f"MAE blocs  : {mae:.4f}"
        )


        results.append(
            {
                "model": model_name,
                "validation": (
                    "Regular spatial blocks"
                ),
                "block_size_km": (
                    BLOCK_SIZE_KM
                ),
                "n_blocks": (
                    n_blocks
                ),
                "n_folds": (
                    N_SPATIAL_FOLDS
                ),
                "n_clusters": (
                    len(df)
                ),
                "n_predictors": (
                    X.shape[1]
                ),
                "R2": r2,
                "RMSE": rmse,
                "MAE": mae,
                "median_alpha": float(
                    np.median(
                        selected_alphas
                    )
                ),
                "min_alpha": float(
                    np.min(
                        selected_alphas
                    )
                ),
                "max_alpha": float(
                    np.max(
                        selected_alphas
                    )
                ),
            }
        )


        safe_name = (
            model_name
            .lower()
            .replace(
                "+",
                "_plus_"
            )
        )


        prediction_table[
            f"pred_{safe_name}"
        ] = predictions


    # ============================================================
    # 19. RESULTATS
    # ============================================================

    results_df = pd.DataFrame(
        results
    )


    results_df = (
        results_df
        .sort_values(
            "RMSE"
        )
        .reset_index(
            drop=True
        )
    )


    print(
        "\n" + "=" * 72
    )

    print(
        "RESULTATS - BLOCS SPATIAUX REGULIERS"
    )

    print(
        "=" * 72
    )


    print(
        results_df[
            [
                "model",
                "R2",
                "RMSE",
                "MAE",
                "median_alpha",
                "n_predictors",
            ]
        ].to_string(
            index=False
        )
    )


    # ============================================================
    # 20. SAUVEGARDE
    # ============================================================

    results_df.to_csv(
        METRICS_FILE,
        index=False
    )


    prediction_table.to_csv(
        PREDICTIONS_FILE,
        index=False
    )


    # ============================================================
    # 21. COMPARAISON DES TROIS VALIDATIONS
    # ============================================================

    random_file = (
        OUTPUT_DIR
        / "ridge_model_comparison.csv"
    )

    kmeans_file = (
        OUTPUT_DIR
        / "spatial_validation_metrics.csv"
    )


    comparison_parts = []


    if random_file.exists():

        random_df = pd.read_csv(
            random_file
        )[
            [
                "model",
                "R2",
                "RMSE",
                "MAE",
            ]
        ].copy()


        random_df[
            "method"
        ] = "Random KFold"


        comparison_parts.append(
            random_df
        )


    if kmeans_file.exists():

        kmeans_df = pd.read_csv(
            kmeans_file
        )[
            [
                "model",
                "R2",
                "RMSE",
                "MAE",
            ]
        ].copy()


        kmeans_df[
            "method"
        ] = "Spatial KMeans groups"


        comparison_parts.append(
            kmeans_df
        )


    block_df = results_df[
        [
            "model",
            "R2",
            "RMSE",
            "MAE",
        ]
    ].copy()


    block_df[
        "method"
    ] = (
        f"Regular blocks "
        f"{BLOCK_SIZE_KM:.0f}km"
    )


    comparison_parts.append(
        block_df
    )


    comparison_df = pd.concat(
        comparison_parts,
        ignore_index=True
    )


    comparison_df.to_csv(
        COMPARISON_FILE,
        index=False
    )


    print(
        "\n" + "=" * 72
    )

    print(
        "COMPARAISON DES METHODES DE VALIDATION"
    )

    print(
        "=" * 72
    )


    print(
        comparison_df[
            [
                "method",
                "model",
                "R2",
                "RMSE",
                "MAE",
            ]
        ].to_string(
            index=False
        )
    )


    # ============================================================
    # 22. FICHIERS PRODUITS
    # ============================================================

    print(
        "\nFichiers sauvegardés :"
    )

    print(
        METRICS_FILE
    )

    print(
        PREDICTIONS_FILE
    )

    print(
        BLOCKS_FILE
    )

    print(
        COMPARISON_FILE
    )


    print(
        "\n" + "=" * 72
    )

    print(
        "ETAPE 08B TERMINEE AVEC SUCCES"
    )

    print(
        "=" * 72
    )

if __name__ == "__main__":
    main()
