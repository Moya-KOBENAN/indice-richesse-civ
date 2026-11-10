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
# 07B - COMPARAISON RIDGE AVEC FEATURES ADAPTEES A VIIRS
# Projet : Indice de richesse - Côte d'Ivoire
#
# Comparaison :
# 1. VIIRS
# 2. CNN ImageNet
# 3. CNN adapté à VIIRS
# 4. CNN ImageNet + VIIRS
# 5. CNN adapté à VIIRS + VIIRS
#
# Validation :
# Nested K-Fold aléatoire
# - Outer CV : 10 folds
# - Inner CV : 5 folds
# ============================================================


# ============================================================
# 1. CHEMINS
# ============================================================


MAIN_FILE = (
    DHS_PROCESSED_DIR / 'DHS_CI_2021_richesse_GPS_VIIRS_merged.csv'
)

IMAGENET_FEATURES_FILE = (
    DHS_PROCESSED_DIR / 'cnn_features_resnet50.csv'
)

VIIRS_FEATURES_FILE = (
    DHS_PROCESSED_DIR / 'cnn_features_resnet50_viirs.csv'
)

OUTPUT_TABLE = (
    DHS_TABLES_DIR / 'ridge_viirs_adapted_comparison.csv'
)

OUTPUT_PREDICTIONS = (
    DHS_TABLES_DIR / 'ridge_viirs_adapted_predictions.csv'
)


# ============================================================
# 2. PARAMETRES
# ============================================================

TARGET = "wealth_mean"

VIIRS_VARIABLE = "viirs_2u5r_mean"

RANDOM_STATE = 42

N_OUTER_SPLITS = 10

N_INNER_SPLITS = 5

ALPHAS = np.logspace(
    -3,
    4,
    20
)


# ============================================================
# 3. CHARGEMENT
# ============================================================

print("=" * 76)
print("07B - RIDGE : COMPARAISON AVANT / APRES ADAPTATION VIIRS")
print("=" * 76)


main_df = pd.read_csv(
    MAIN_FILE
)

imagenet_df = pd.read_csv(
    IMAGENET_FEATURES_FILE
)

adapted_df = pd.read_csv(
    VIIRS_FEATURES_FILE
)


print(
    f"Base principale : {main_df.shape}"
)

print(
    f"Features ImageNet : {imagenet_df.shape}"
)

print(
    f"Features adaptées VIIRS : {adapted_df.shape}"
)


# ============================================================
# 4. CONTROLES
# ============================================================

for name, dataframe in {
    "main": main_df,
    "imagenet": imagenet_df,
    "adapted": adapted_df,
}.items():

    if "cluster" not in dataframe.columns:

        raise ValueError(
            f"Colonne cluster absente dans {name}"
        )

    if dataframe["cluster"].duplicated().any():

        raise ValueError(
            f"Clusters dupliqués dans {name}"
        )


if TARGET not in main_df.columns:

    raise ValueError(
        f"Cible absente : {TARGET}"
    )


if VIIRS_VARIABLE not in main_df.columns:

    raise ValueError(
        f"VIIRS absent : {VIIRS_VARIABLE}"
    )


# ============================================================
# 5. MERGE
# ============================================================

df = (
    main_df
    .merge(
        imagenet_df,
        on="cluster",
        how="inner"
    )
    .merge(
        adapted_df,
        on="cluster",
        how="inner"
    )
)


print(
    f"Base après fusion : {df.shape}"
)


if len(df) != 539:

    raise ValueError(
        f"539 clusters attendus, {len(df)} obtenus."
    )


# ============================================================
# 6. VARIABLES
# ============================================================

imagenet_columns = [
    c
    for c in imagenet_df.columns
    if c.startswith("cnn_")
]


adapted_columns = [
    c
    for c in adapted_df.columns
    if c.startswith("cnn_viirs_")
]


print(
    f"Features ImageNet : {len(imagenet_columns)}"
)

print(
    f"Features adaptées : {len(adapted_columns)}"
)


if len(imagenet_columns) != 2048:

    raise ValueError(
        "2048 features ImageNet attendues."
    )


if len(adapted_columns) != 2048:

    raise ValueError(
        "2048 features adaptées attendues."
    )


# Transformation VIIRS identique
# à celle utilisée précédemment.

df["viirs_log"] = np.log1p(
    df[VIIRS_VARIABLE]
)


# ============================================================
# 7. CIBLE
# ============================================================

y = (
    df[TARGET]
    .to_numpy(dtype=float)
)


if not np.isfinite(y).all():

    raise ValueError(
        "La cible contient des valeurs invalides."
    )


# ============================================================
# 8. MODELES A COMPARER
# ============================================================

model_sets = {

    "VIIRS":
        ["viirs_log"],

    "CNN_ImageNet":
        imagenet_columns,

    "CNN_VIIRS_adapte":
        adapted_columns,

    "CNN_ImageNet_plus_VIIRS":
        imagenet_columns
        + ["viirs_log"],

    "CNN_VIIRS_adapte_plus_VIIRS":
        adapted_columns
        + ["viirs_log"],
}


# ============================================================
# 9. PIPELINE RIDGE
# ============================================================

def build_pipeline():

    return Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
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
# 10. NESTED CROSS-VALIDATION
# ============================================================

outer_cv = KFold(
    n_splits=N_OUTER_SPLITS,
    shuffle=True,
    random_state=RANDOM_STATE
)


results = []

prediction_rows = []


print()
print("=" * 76)
print("DEBUT DE LA VALIDATION CROISEE")
print("=" * 76)


for model_name, columns in model_sets.items():

    print()
    print("-" * 76)
    print(
        f"MODELE : {model_name}"
    )
    print(
        f"Nombre de prédicteurs : {len(columns)}"
    )
    print("-" * 76)


    X = (
        df[columns]
        .to_numpy(dtype=float)
    )


    oof_predictions = np.full(
        len(df),
        np.nan,
        dtype=float
    )


    fold_alphas = []


    for fold, (
        train_index,
        test_index
    ) in enumerate(
        outer_cv.split(X),
        start=1
    ):

        X_train = X[
            train_index
        ]

        X_test = X[
            test_index
        ]

        y_train = y[
            train_index
        ]


        inner_cv = KFold(
            n_splits=N_INNER_SPLITS,
            shuffle=True,
            random_state=(
                RANDOM_STATE
                + fold
            )
        )


        grid = GridSearchCV(
            estimator=build_pipeline(),
            param_grid={
                "ridge__alpha":
                    ALPHAS
            },
            scoring=(
                "neg_root_mean_squared_error"
            ),
            cv=inner_cv,
            n_jobs=-1,
        )


        grid.fit(
            X_train,
            y_train
        )


        best_alpha = (
            grid.best_params_[
                "ridge__alpha"
            ]
        )


        fold_alphas.append(
            best_alpha
        )


        predictions = (
            grid.best_estimator_
            .predict(
                X_test
            )
        )


        oof_predictions[
            test_index
        ] = predictions


        print(
            f"Fold {fold:02d}/{N_OUTER_SPLITS} | "
            f"alpha = {best_alpha:.6f}"
        )


    # --------------------------------------------------------
    # Vérification OOF
    # --------------------------------------------------------

    if np.isnan(
        oof_predictions
    ).any():

        raise ValueError(
            f"Prédictions manquantes pour {model_name}"
        )


    # --------------------------------------------------------
    # Métriques globales OOF
    # --------------------------------------------------------

    r2 = r2_score(
        y,
        oof_predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y,
            oof_predictions
        )
    )

    mae = mean_absolute_error(
        y,
        oof_predictions
    )

    median_alpha = float(
        np.median(
            fold_alphas
        )
    )


    results.append(
        {
            "model": model_name,
            "r2": r2,
            "rmse": rmse,
            "mae": mae,
            "median_alpha":
                median_alpha,
            "n_predictors":
                len(columns),
            "validation":
                "nested_random_kfold",
        }
    )


    for i in range(
        len(df)
    ):

        prediction_rows.append(
            {
                "cluster":
                    int(
                        df.iloc[i][
                            "cluster"
                        ]
                    ),

                "model":
                    model_name,

                "observed":
                    float(y[i]),

                "predicted":
                    float(
                        oof_predictions[i]
                    ),

                "error":
                    float(
                        oof_predictions[i]
                        - y[i]
                    ),
            }
        )


    print()
    print(
        f"R²   : {r2:.6f}"
    )

    print(
        f"RMSE : {rmse:.6f}"
    )

    print(
        f"MAE  : {mae:.6f}"
    )


# ============================================================
# 11. TABLEAU FINAL
# ============================================================

results_df = pd.DataFrame(
    results
)


# Tri uniquement pour faciliter la lecture.
# Ne pas interpréter comme classement définitif.

results_df = (
    results_df
    .sort_values(
        "r2",
        ascending=False
    )
    .reset_index(drop=True)
)


predictions_df = pd.DataFrame(
    prediction_rows
)


OUTPUT_TABLE.parent.mkdir(
    parents=True,
    exist_ok=True
)


results_df.to_csv(
    OUTPUT_TABLE,
    index=False
)


predictions_df.to_csv(
    OUTPUT_PREDICTIONS,
    index=False
)


# ============================================================
# 12. BILAN
# ============================================================

print()
print("=" * 76)
print("RESULTATS FINAUX - VALIDATION ALEATOIRE")
print("=" * 76)

print(
    results_df.to_string(
        index=False
    )
)

print()
print(
    f"Tableau sauvegardé : {OUTPUT_TABLE}"
)

print(
    f"Prédictions sauvegardées : "
    f"{OUTPUT_PREDICTIONS}"
)

print()
print("=" * 76)
print("07B TERMINE AVEC SUCCES")
print("=" * 76)