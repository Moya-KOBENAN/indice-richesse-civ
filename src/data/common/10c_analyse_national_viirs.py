"""Étape historique isolée : exécution explicite uniquement via main()."""


def main():
    # Résolution des chemins indépendante du répertoire de lancement.
    from pathlib import Path
    import sys

    _PROJECT_ROOT = Path(__file__).resolve().parents[3]
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))

    from src.utils.paths import (
        COMMON_FIGURES_DIR,
        COMMON_PROCESSED_DIR,
        COMMON_TABLES_DIR,
        PROJECT_ROOT,
        ensure_layout,
    )

    ensure_layout('common')


    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt


    # ============================================================
    # 10C - DIAGNOSTIC DE LA CIBLE VIIRS NATIONALE
    # ============================================================


    INPUT_FILE = (
        COMMON_PROCESSED_DIR / 'cote_ivoire_national_grid_viirs.csv'
    )

    OUTPUT_TABLE = (
        COMMON_TABLES_DIR / 'national_viirs_diagnostic.csv'
    )

    OUTPUT_FIGURE = (
        COMMON_FIGURES_DIR / 'national_viirs_distribution.png'
    )

    EXPECTED_POINTS = 3218


    # ============================================================
    # 1. CHARGEMENT
    # ============================================================

    print("=" * 72)
    print("10C - DIAGNOSTIC DE LA CIBLE VIIRS NATIONALE")
    print("=" * 72)

    df = pd.read_csv(INPUT_FILE)

    print(f"Dimensions : {df.shape}")


    required_columns = [
        "grid_id",
        "longitude",
        "latitude",
        "viirs_mean_2p5km",
    ]

    for column in required_columns:
        if column not in df.columns:
            raise ValueError(
                f"Colonne absente : {column}"
            )


    if len(df) != EXPECTED_POINTS:
        raise ValueError(
            f"{EXPECTED_POINTS} points attendus, "
            f"{len(df)} trouvés."
        )


    # ============================================================
    # 2. CONTROLES
    # ============================================================

    viirs = df["viirs_mean_2p5km"].astype(float)

    n_total = len(df)
    n_missing = int(viirs.isna().sum())
    n_negative = int((viirs < 0).sum())
    n_zero = int((viirs == 0).sum())
    n_positive = int((viirs > 0).sum())

    print()
    print("=" * 72)
    print("CONTROLE GENERAL")
    print("=" * 72)

    print(f"Total : {n_total}")
    print(f"Manquants : {n_missing}")
    print(f"Négatifs : {n_negative}")
    print(f"Zéros : {n_zero}")
    print(f"Positifs : {n_positive}")

    print(
        f"Part des zéros : "
        f"{100 * n_zero / n_total:.2f}%"
    )

    print(
        f"Part des positifs : "
        f"{100 * n_positive / n_total:.2f}%"
    )


    # ============================================================
    # 3. DISTRIBUTION COMPLETE
    # ============================================================

    print()
    print("=" * 72)
    print("DISTRIBUTION COMPLETE")
    print("=" * 72)

    quantiles_all = viirs.quantile(
        [
            0.00,
            0.10,
            0.25,
            0.50,
            0.75,
            0.90,
            0.95,
            0.99,
            1.00,
        ]
    )

    print(quantiles_all)


    # ============================================================
    # 4. DISTRIBUTION DES VALEURS POSITIVES
    # ============================================================

    positive = viirs[viirs > 0]

    print()
    print("=" * 72)
    print("DISTRIBUTION DES VALEURS STRICTEMENT POSITIVES")
    print("=" * 72)

    if len(positive) == 0:
        raise ValueError(
            "Aucune valeur VIIRS positive."
        )


    quantile_levels = [
        0.00,
        0.10,
        0.25,
        0.50,
        0.75,
        0.90,
        0.95,
        0.99,
        1.00,
    ]

    quantiles_positive = positive.quantile(
        quantile_levels
    )

    print(quantiles_positive)


    # ============================================================
    # 5. TRANSFORMATION LOG1P
    # ============================================================

    if n_negative > 0:
        raise ValueError(
            "Des valeurs VIIRS négatives sont présentes."
        )


    df["viirs_log1p"] = np.log1p(
        viirs
    )


    print()
    print("=" * 72)
    print("DISTRIBUTION LOG1P")
    print("=" * 72)

    print(
        df["viirs_log1p"].describe(
            percentiles=[
                0.10,
                0.25,
                0.50,
                0.75,
                0.90,
                0.95,
                0.99,
            ]
        )
    )


    # ============================================================
    # 6. CLASSES EXPERIMENTALES
    # ============================================================
    #
    # Classe 0 :
    # aucune luminosité détectée
    #
    # Classes 1 à 4 :
    # quartiles calculés uniquement parmi
    # les valeurs strictement positives.
    #
    # Ces classes servent uniquement au diagnostic.
    # Elles ne sont pas encore la cible définitive du CNN.
    # ============================================================

    q25 = positive.quantile(0.25)
    q50 = positive.quantile(0.50)
    q75 = positive.quantile(0.75)


    def assign_light_class(value):

        if value <= 0:
            return 0

        if value <= q25:
            return 1

        if value <= q50:
            return 2

        if value <= q75:
            return 3

        return 4


    df["viirs_class_5"] = (
        viirs.apply(assign_light_class)
    )


    class_counts = (
        df["viirs_class_5"]
        .value_counts()
        .sort_index()
    )


    print()
    print("=" * 72)
    print("CLASSES EXPERIMENTALES DE LUMINOSITE")
    print("=" * 72)

    for class_id, count in class_counts.items():

        percentage = (
            100 * count / n_total
        )

        print(
            f"Classe {class_id} : "
            f"{count} points "
            f"({percentage:.2f}%)"
        )


    print()
    print("Seuils des valeurs positives :")
    print(f"Q25 : {q25:.6f}")
    print(f"Q50 : {q50:.6f}")
    print(f"Q75 : {q75:.6f}")


    # ============================================================
    # 7. TABLEAU RECAPITULATIF
    # ============================================================

    summary = pd.DataFrame(
        {
            "indicator": [
                "n_total",
                "n_missing",
                "n_negative",
                "n_zero",
                "n_positive",
                "pct_zero",
                "pct_positive",
                "positive_q25",
                "positive_q50",
                "positive_q75",
                "positive_q90",
                "positive_q95",
                "positive_q99",
                "maximum",
            ],

            "value": [
                n_total,
                n_missing,
                n_negative,
                n_zero,
                n_positive,
                100 * n_zero / n_total,
                100 * n_positive / n_total,
                positive.quantile(0.25),
                positive.quantile(0.50),
                positive.quantile(0.75),
                positive.quantile(0.90),
                positive.quantile(0.95),
                positive.quantile(0.99),
                viirs.max(),
            ],
        }
    )


    OUTPUT_TABLE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    summary.to_csv(
        OUTPUT_TABLE,
        index=False
    )


    # ============================================================
    # 8. FIGURE
    # ============================================================

    OUTPUT_FIGURE.parent.mkdir(
        parents=True,
        exist_ok=True
    )


    plt.figure(
        figsize=(9, 6)
    )

    plt.hist(
        df["viirs_log1p"],
        bins=50
    )

    plt.xlabel(
        "log(1 + luminosité VIIRS)"
    )

    plt.ylabel(
        "Nombre de points"
    )

    plt.title(
        "Distribution de la luminosité nocturne\n"
        "Grille nationale de Côte d'Ivoire"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_FIGURE,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()


    # ============================================================
    # 9. SAUVEGARDE DES DONNEES DIAGNOSTIQUES
    # ============================================================

    DIAGNOSTIC_DATA_FILE = (
        COMMON_PROCESSED_DIR / 'cote_ivoire_national_grid_viirs_diagnostic.csv'
    )

    df.to_csv(
        DIAGNOSTIC_DATA_FILE,
        index=False
    )


    # ============================================================
    # 10. FIN
    # ============================================================

    print()
    print("=" * 72)
    print("FICHIERS CREES")
    print("=" * 72)

    print(
        f"Résumé : {OUTPUT_TABLE}"
    )

    print(
        f"Figure : {OUTPUT_FIGURE}"
    )

    print(
        f"Données diagnostic : "
        f"{DIAGNOSTIC_DATA_FILE}"
    )

    print()
    print("=" * 72)
    print("10C TERMINE AVEC SUCCES")
    print("=" * 72)

if __name__ == "__main__":
    main()
