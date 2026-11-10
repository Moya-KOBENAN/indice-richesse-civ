"""Étape historique isolée : exécution explicite uniquement via main()."""


def main():
    # Résolution des chemins indépendante du répertoire de lancement.
    from pathlib import Path
    import sys

    _PROJECT_ROOT = Path(__file__).resolve().parents[3]
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))

    from src.utils.paths import (
        DHS_FIGURES_DIR,
        DHS_PROCESSED_DIR,
        DHS_TABLES_DIR,
        PROJECT_ROOT,
        ensure_layout,
    )

    ensure_layout('dhs')

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    from sklearn.model_selection import KFold, cross_val_predict
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


    # ---------------------------------------------------------
    # 1. CHEMINS
    # ---------------------------------------------------------

    ROOT = PROJECT_ROOT

    DATA_PROCESSED = DHS_PROCESSED_DIR
    OUTPUT_TABLES = DHS_TABLES_DIR
    OUTPUT_FIGURES = DHS_FIGURES_DIR

    OUTPUT_TABLES.mkdir(parents=True, exist_ok=True)
    OUTPUT_FIGURES.mkdir(parents=True, exist_ok=True)


    # ---------------------------------------------------------
    # 2. CHARGEMENT DES DONNEES
    # ---------------------------------------------------------

    input_file = (
        DATA_PROCESSED
        / "DHS_CI_2021_richesse_GPS_VIIRS_merged.csv"
    )

    df = pd.read_csv(input_file)

    print("Dimensions de la base :", df.shape)


    # ---------------------------------------------------------
    # 3. VARIABLE CIBLE
    # ---------------------------------------------------------

    target = "wealth_mean"

    y = df[target].to_numpy()


    # ---------------------------------------------------------
    # 4. VARIABLES VIIRS A TESTER
    # ---------------------------------------------------------

    predictors = {
        "2 km urbain / 5 km rural": "viirs_2u5r_mean",
        "5 km pour tous": "viirs_5all_mean",
        "2 km urbain / 10 km rural": "viirs_2u10r_mean",
    }


    # ---------------------------------------------------------
    # 5. VALIDATION CROISEE
    # ---------------------------------------------------------

    kf = KFold(
        n_splits=10,
        shuffle=True,
        random_state=42
    )

    results = []
    predictions = {}


    # ---------------------------------------------------------
    # 6. MODELES BASELINE
    # ---------------------------------------------------------

    for label, variable in predictors.items():

        X_raw = df[[variable]].to_numpy()

        # -----------------------------
        # MODELE 1 : radiance brute
        # -----------------------------

        model_raw = LinearRegression()

        pred_raw = cross_val_predict(
            model_raw,
            X_raw,
            y,
            cv=kf
        )

        results.append({
            "buffer": label,
            "transformation": "radiance brute",
            "R2_CV": r2_score(y, pred_raw),
            "RMSE_CV": np.sqrt(
                mean_squared_error(y, pred_raw)
            ),
            "MAE_CV": mean_absolute_error(
                y,
                pred_raw
            ),
            "n_clusters": len(y)
        })

        predictions[
            (label, "radiance brute")
        ] = pred_raw


        # -----------------------------
        # MODELE 2 : log(1 + radiance)
        # -----------------------------

        X_log = np.log1p(X_raw)

        model_log = LinearRegression()

        pred_log = cross_val_predict(
            model_log,
            X_log,
            y,
            cv=kf
        )

        results.append({
            "buffer": label,
            "transformation": "log(1 + radiance)",
            "R2_CV": r2_score(y, pred_log),
            "RMSE_CV": np.sqrt(
                mean_squared_error(y, pred_log)
            ),
            "MAE_CV": mean_absolute_error(
                y,
                pred_log
            ),
            "n_clusters": len(y)
        })

        predictions[
            (label, "log(1 + radiance)")
        ] = pred_log


    # ---------------------------------------------------------
    # 7. TABLEAU DES RESULTATS
    # ---------------------------------------------------------

    metrics = pd.DataFrame(results)

    metrics = metrics.sort_values(
        "R2_CV",
        ascending=False
    )

    print("\nRésultats du benchmark VIIRS :")
    print(metrics.to_string(index=False))


    # ---------------------------------------------------------
    # 8. SAUVEGARDE DES METRIQUES
    # ---------------------------------------------------------

    metrics_file = (
        OUTPUT_TABLES
        / "VIIRS_baseline_metrics.csv"
    )

    metrics.to_csv(
        metrics_file,
        index=False
    )

    print("\nTableau sauvegardé :")
    print(metrics_file)


    # ---------------------------------------------------------
    # 9. MEILLEURE SPECIFICATION
    # ---------------------------------------------------------

    best = metrics.iloc[0]

    best_key = (
        best["buffer"],
        best["transformation"]
    )

    best_pred = predictions[best_key]

    print("\nMeilleure spécification :")
    print(best)


    # ---------------------------------------------------------
    # 10. GRAPHIQUE OBSERVE VS PREDIT
    # ---------------------------------------------------------

    plt.figure(figsize=(7, 6))

    plt.scatter(
        y,
        best_pred,
        alpha=0.65
    )

    min_val = min(
        y.min(),
        best_pred.min()
    )

    max_val = max(
        y.max(),
        best_pred.max()
    )

    plt.plot(
        [min_val, max_val],
        [min_val, max_val],
        linestyle="--"
    )

    plt.xlabel(
        "Richesse observée"
    )

    plt.ylabel(
        "Richesse prédite par VIIRS"
    )

    plt.title(
        f"Baseline VIIRS\n"
        f"{best['buffer']} - "
        f"{best['transformation']}"
    )

    plt.tight_layout()


    figure_file = (
        OUTPUT_FIGURES
        / "VIIRS_observe_vs_pred.png"
    )

    plt.savefig(
        figure_file,
        dpi=180
    )

    plt.close()

    print("\nGraphique sauvegardé :")
    print(figure_file)

if __name__ == "__main__":
    main()
