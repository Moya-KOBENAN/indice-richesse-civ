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
        KFold,
    )
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler


    # ============================================================
    # 07 - MODELISATION RIDGE
    # Projet : Estimation spatiale de la richesse
    # Côte d'Ivoire - DHS 2021
    #
    # Comparaison exploratoire :
    # 1. VIIRS seul
    # 2. CNN seul
    # 3. CNN + VIIRS
    #
    # La validation spatiale finale sera réalisée à l'étape 08.
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
        / "ridge_model_comparison.csv"
    )

    PREDICTIONS_FILE = (
        OUTPUT_DIR
        / "ridge_predictions.csv"
    )

    TARGET = "wealth_mean"

    RANDOM_STATE = 42

    N_SPLITS = 10

    # Même transformation VIIRS que notre meilleur benchmark
    VIIRS_VARIABLE = "viirs_2u5r_mean"

    # Valeurs de pénalisation Ridge à tester
    ALPHAS = np.logspace(
        -3,
        4,
        20
    )


    # ============================================================
    # 2. CHARGEMENT DES DONNEES
    # ============================================================

    print("=" * 65)
    print("ETAPE 07 - MODELISATION RIDGE")
    print("=" * 65)

    main_df = pd.read_csv(
        MAIN_FILE
    )

    cnn_df = pd.read_csv(
        CNN_FILE
    )

    print(
        f"Base principale : "
        f"{main_df.shape}"
    )

    print(
        f"Base CNN : "
        f"{cnn_df.shape}"
    )


    # ============================================================
    # 3. CONTROLES
    # ============================================================

    required_main_columns = [
        "cluster",
        TARGET,
        VIIRS_VARIABLE,
    ]

    for column in required_main_columns:

        if column not in main_df.columns:

            raise ValueError(
                f"Variable absente de la base principale : "
                f"{column}"
            )


    if "cluster" not in cnn_df.columns:

        raise ValueError(
            "La variable cluster est absente "
            "de la base CNN."
        )


    if main_df["cluster"].duplicated().any():

        raise ValueError(
            "Doublons de clusters dans la base principale."
        )


    if cnn_df["cluster"].duplicated().any():

        raise ValueError(
            "Doublons de clusters dans la base CNN."
        )


    # ============================================================
    # 4. FUSION CNN + DHS + VIIRS
    # ============================================================

    df = main_df.merge(
        cnn_df,
        on="cluster",
        how="inner",
        validate="one_to_one",
    )

    print(
        f"Base après fusion : "
        f"{df.shape}"
    )


    if len(df) != 539:

        raise ValueError(
            f"539 clusters attendus après fusion, "
            f"{len(df)} obtenus."
        )


    # ============================================================
    # 5. VARIABLE CIBLE
    # ============================================================

    y = df[TARGET].to_numpy(
        dtype=float
    )


    if not np.isfinite(y).all():

        raise ValueError(
            "La variable cible contient "
            "des valeurs manquantes ou infinies."
        )


    # ============================================================
    # 6. VARIABLES CNN
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
    # 7. PREPARATION DES TROIS MODELES
    # ============================================================

    # VIIRS :
    # transformation log(1 + radiance)
    # retenue lors du benchmark précédent.

    viirs_log = np.log1p(
        df[
            VIIRS_VARIABLE
        ].clip(lower=0)
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


    print(
        f"Nombre de variables CNN : "
        f"{len(cnn_columns)}"
    )

    print(
        "Modèles comparés : "
        "VIIRS / CNN / CNN+VIIRS"
    )


    # ============================================================
    # 8. VALIDATION CROISEE EXTERNE
    # ============================================================

    # Les mêmes plis sont utilisés pour les trois modèles.
    # Cela rend leur comparaison plus cohérente.

    outer_cv = KFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    splits = list(
        outer_cv.split(df)
    )


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
    # 10. BOUCLE DE MODELISATION
    # ============================================================

    results = []

    prediction_table = pd.DataFrame(
        {
            "cluster": df["cluster"].to_numpy(),
            "wealth_observed": y,
        }
    )


    for model_name, X in datasets.items():

        print("\n" + "-" * 65)

        print(
            f"MODELE : {model_name}"
        )

        print("-" * 65)

        predictions = np.full(
            len(df),
            np.nan
        )

        selected_alphas = []


        # --------------------------------------------------------
        # Validation externe
        # --------------------------------------------------------

        for fold_number, (
            train_index,
            test_index
        ) in enumerate(
            splits,
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
            # Validation interne pour choisir alpha
            # ----------------------------------------------------

            inner_cv = KFold(
                n_splits=5,
                shuffle=True,
                random_state=(
                    RANDOM_STATE
                    + fold_number
                ),
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
                n_jobs=-1,
            )


            search.fit(
                X_train,
                y_train
            )


            best_model = (
                search.best_estimator_
            )


            best_alpha = (
                search.best_params_[
                    "ridge__alpha"
                ]
            )


            selected_alphas.append(
                float(best_alpha)
            )


            # ----------------------------------------------------
            # Prédiction uniquement sur le pli test
            # ----------------------------------------------------

            predictions[
                test_index
            ] = best_model.predict(
                X_test
            )


            print(
                f"Fold {fold_number:02d}/{N_SPLITS} "
                f"| alpha = {best_alpha:.6g}"
            )


        # ========================================================
        # 11. METRIQUES OUT-OF-FOLD
        # ========================================================

        if np.isnan(
            predictions
        ).any():

            raise ValueError(
                f"Certaines prédictions sont "
                f"manquantes pour {model_name}."
            )


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
            f"\nR²   : {r2:.4f}"
        )

        print(
            f"RMSE : {rmse:.4f}"
        )

        print(
            f"MAE  : {mae:.4f}"
        )


        results.append(
            {
                "model": model_name,
                "validation": (
                    "Nested KFold aleatoire"
                ),
                "outer_folds": N_SPLITS,
                "inner_folds": 5,
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


        prediction_table[
            f"pred_{model_name.lower().replace('+', '_plus_')}"
        ] = predictions


    # ============================================================
    # 12. TABLEAU COMPARATIF
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


    print("\n" + "=" * 65)

    print(
        "COMPARAISON DES MODELES"
    )

    print("=" * 65)


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
    # 13. SAUVEGARDE
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
        "\nRésultats sauvegardés :"
    )

    print(
        METRICS_FILE
    )

    print(
        PREDICTIONS_FILE
    )


    # ============================================================
    # 14. RAPPEL METHODOLOGIQUE
    # ============================================================

    print("\n" + "=" * 65)

    print(
        "IMPORTANT"
    )

    print("=" * 65)

    print(
        "Ces résultats utilisent une validation croisée "
        "aléatoire imbriquée."
    )

    print(
        "Ils constituent une comparaison exploratoire."
    )

    print(
        "La performance finale sera évaluée avec une "
        "validation spatiale à l'étape 08."
    )


    print("\n" + "=" * 65)

    print(
        "ETAPE 07 TERMINEE AVEC SUCCES"
    )

    print("=" * 65)

if __name__ == "__main__":
    main()
