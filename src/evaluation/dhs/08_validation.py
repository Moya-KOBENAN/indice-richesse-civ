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
    from sklearn.cluster import KMeans
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
    # 08 - VALIDATION SPATIALE
    # Projet : Estimation spatiale de la richesse
    # Côte d'Ivoire - DHS 2021
    #
    # Objectif :
    # évaluer la capacité des modèles à prédire la richesse
    # dans des zones géographiques non utilisées à l'entraînement.
    #
    # Modèles :
    # 1. VIIRS
    # 2. CNN
    # 3. CNN + VIIRS
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
        / "spatial_validation_metrics.csv"
    )

    PREDICTIONS_FILE = (
        OUTPUT_DIR
        / "spatial_validation_predictions.csv"
    )

    GROUPS_FILE = (
        OUTPUT_DIR
        / "spatial_validation_groups.csv"
    )

    TARGET = "wealth_mean"

    VIIRS_VARIABLE = "viirs_2u5r_mean"

    RANDOM_STATE = 42

    # Nombre de groupes géographiques créés
    N_SPATIAL_GROUPS = 10

    # Nombre de plis de validation spatiale
    N_SPATIAL_FOLDS = 5

    # Grille Ridge
    ALPHAS = np.logspace(
        -3,
        4,
        20
    )


    # ============================================================
    # 2. CHARGEMENT DES DONNEES
    # ============================================================

    print("=" * 70)
    print("ETAPE 08 - VALIDATION SPATIALE")
    print("=" * 70)

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
    # 3. CONTROLES PRELIMINAIRES
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


    print(
        f"Base fusionnée : {df.shape}"
    )


    # ============================================================
    # 5. CREATION DES GROUPES SPATIAUX
    # ============================================================

    # Pour créer des groupes géographiques, on convertit
    # approximativement latitude/longitude en kilomètres.
    #
    # Cette transformation sert uniquement à constituer
    # les groupes spatiaux, pas comme variable prédictive.

    mean_latitude = np.radians(
        df["latitude"].mean()
    )

    x_km = (
        df["longitude"].to_numpy()
        * 111.32
        * np.cos(mean_latitude)
    )

    y_km = (
        df["latitude"].to_numpy()
        * 110.57
    )

    coordinates_km = np.column_stack(
        [
            x_km,
            y_km,
        ]
    )


    # KMeans sert uniquement à regrouper
    # des clusters géographiquement proches.

    spatial_clusterer = KMeans(
        n_clusters=N_SPATIAL_GROUPS,
        random_state=RANDOM_STATE,
        n_init=20,
    )

    spatial_groups = (
        spatial_clusterer
        .fit_predict(
            coordinates_km
        )
    )


    df["spatial_group"] = spatial_groups


    print(
        f"Nombre de groupes spatiaux : "
        f"{df['spatial_group'].nunique()}"
    )

    print("\nTaille des groupes :")

    print(
        df[
            "spatial_group"
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )


    # ============================================================
    # 6. SAUVEGARDE DES GROUPES
    # ============================================================

    group_table = df[
        [
            "cluster",
            "latitude",
            "longitude",
            "spatial_group",
        ]
    ].copy()

    group_table.to_csv(
        GROUPS_FILE,
        index=False
    )


    # ============================================================
    # 7. VARIABLE CIBLE
    # ============================================================

    y = df[
        TARGET
    ].to_numpy(
        dtype=float
    )


    if not np.isfinite(y).all():

        raise ValueError(
            "Valeurs invalides dans la cible."
        )


    # ============================================================
    # 8. VARIABLES EXPLICATIVES
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


    # VIIRS transformé comme à l'étape 07

    viirs_log = np.log1p(
        df[
            VIIRS_VARIABLE
        ].clip(
            lower=0
        )
    )


    X_viirs = pd.DataFrame(
        {
            "log_viirs": viirs_log
        }
    )


    X_cnn = df[
        cnn_columns
    ].copy()


    X_cnn_viirs = X_cnn.copy()

    X_cnn_viirs[
        "log_viirs"
    ] = viirs_log.to_numpy()


    datasets = {
        "VIIRS": X_viirs,
        "CNN": X_cnn,
        "CNN+VIIRS": X_cnn_viirs,
    }


    # ============================================================
    # 9. PIPELINE RIDGE
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
    # 10. VALIDATION SPATIALE EXTERNE
    # ============================================================

    groups = df[
        "spatial_group"
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
        f"\nNombre de plis spatiaux : "
        f"{len(spatial_splits)}"
    )


    # ============================================================
    # 11. VERIFICATION ABSENCE DE FUITE SPATIALE
    # ============================================================

    print(
        "\nContrôle train/test des groupes :"
    )


    for fold_number, (
        train_index,
        test_index
    ) in enumerate(
        spatial_splits,
        start=1
    ):

        train_groups = set(
            groups[
                train_index
            ]
        )

        test_groups = set(
            groups[
                test_index
            ]
        )

        overlap = (
            train_groups
            .intersection(
                test_groups
            )
        )

        print(
            f"Fold {fold_number}: "
            f"train={len(train_index)} "
            f"| test={len(test_index)} "
            f"| groupes communs={len(overlap)}"
        )


        if len(overlap) != 0:

            raise ValueError(
                "Fuite spatiale détectée entre "
                "l'entraînement et le test."
            )


    # ============================================================
    # 12. MODELISATION
    # ============================================================

    results = []


    prediction_table = df[
        [
            "cluster",
            "latitude",
            "longitude",
            "spatial_group",
        ]
    ].copy()


    prediction_table[
        "wealth_observed"
    ] = y


    for model_name, X in datasets.items():

        print(
            "\n" + "-" * 70
        )

        print(
            f"MODELE : {model_name}"
        )

        print(
            "-" * 70
        )


        predictions = np.full(
            len(df),
            np.nan
        )

        selected_alphas = []


        # --------------------------------------------------------
        # Boucle externe spatiale
        # --------------------------------------------------------

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
            # CV interne uniquement sur les données d'entraînement
            # pour choisir alpha.
            #
            # Ici, l'objectif est de ne jamais utiliser
            # les observations du test spatial pour choisir alpha.
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

                # On limite volontairement le parallélisme
                # pour éviter les avertissements mémoire
                # observés à l'étape 07 sur le Mac.
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
        # 13. CONTROLE DES PREDICTIONS
        # ========================================================

        if np.isnan(
            predictions
        ).any():

            raise ValueError(
                f"Prédictions manquantes "
                f"pour {model_name}."
            )


        # ========================================================
        # 14. METRIQUES SPATIALES
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
            f"\nR² spatial   : {r2:.4f}"
        )

        print(
            f"RMSE spatial : {rmse:.4f}"
        )

        print(
            f"MAE spatial  : {mae:.4f}"
        )


        results.append(
            {
                "model": model_name,
                "validation": (
                    "GroupKFold spatial"
                ),
                "n_spatial_groups": (
                    N_SPATIAL_GROUPS
                ),
                "n_folds": (
                    N_SPATIAL_FOLDS
                ),
                "n_clusters": len(df),
                "n_predictors": X.shape[1],
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
    # 15. TABLEAU COMPARATIF
    # ============================================================

    results_df = pd.DataFrame(
        results
    )


    results_df = (
        results_df
        .sort_values(
            "RMSE",
            ascending=True
        )
        .reset_index(
            drop=True
        )
    )


    print(
        "\n" + "=" * 70
    )

    print(
        "RESULTATS DE LA VALIDATION SPATIALE"
    )

    print(
        "=" * 70
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
    # 16. COMPARAISON AVEC ETAPE 07
    # ============================================================

    RANDOM_METRICS_FILE = (
        OUTPUT_DIR
        / "ridge_model_comparison.csv"
    )


    if RANDOM_METRICS_FILE.exists():

        random_results = pd.read_csv(
            RANDOM_METRICS_FILE
        )


        comparison = (
            random_results[
                [
                    "model",
                    "R2",
                    "RMSE",
                    "MAE",
                ]
            ]
            .merge(
                results_df[
                    [
                        "model",
                        "R2",
                        "RMSE",
                        "MAE",
                    ]
                ],
                on="model",
                suffixes=(
                    "_random",
                    "_spatial",
                ),
            )
        )


        comparison[
            "delta_R2"
        ] = (
            comparison[
                "R2_spatial"
            ]
            - comparison[
                "R2_random"
            ]
        )


        comparison[
            "delta_RMSE"
        ] = (
            comparison[
                "RMSE_spatial"
            ]
            - comparison[
                "RMSE_random"
            ]
        )


        comparison_file = (
            OUTPUT_DIR
            / "random_vs_spatial_validation.csv"
        )


        comparison.to_csv(
            comparison_file,
            index=False
        )


        print(
            "\nComparaison validation "
            "aléatoire / spatiale :"
        )


        print(
            comparison.to_string(
                index=False
            )
        )


    # ============================================================
    # 17. SAUVEGARDE
    # ============================================================

    results_df.to_csv(
        METRICS_FILE,
        index=False
    )


    prediction_table.to_csv(
        PREDICTIONS_FILE,
        index=False
    )


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
        GROUPS_FILE
    )


    # ============================================================
    # 18. FIN
    # ============================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "ETAPE 08 TERMINEE AVEC SUCCES"
    )

    print(
        "=" * 70
    )

    print(
        "\nAttention : les performances spatiales "
        "doivent être interprétées comme une estimation "
        "de la généralisation géographique."
    )

    print(
        "La localisation DHS étant déplacée pour préserver "
        "la confidentialité, les résultats ne doivent pas "
        "être interprétés comme des prédictions à l'adresse "
        "exacte des ménages."
    )

if __name__ == "__main__":
    main()
