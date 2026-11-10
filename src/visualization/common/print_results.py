"""Afficher tous les résultats dans le terminal, sans rien recalculer.

Usage :
    python run_pipeline.py --resultats                 # tout
    python run_pipeline.py --resultats --section dhs   # une partie

Sections : donnees, dhs, ehcvm, externe, articles, carte.
Lit uniquement les tables déjà produites dans outputs/ et data/processed/.
Une section dont les fichiers manquent l'indique au lieu d'échouer.
"""
from pathlib import Path
import argparse
import sys

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np
import pandas as pd

from src.utils.config import load_config
from src.utils.paths import (COMMON_TABLES_DIR, COMPARISON_OUTPUTS_DIR, EHCVM_TABLES_DIR,
                             PROJECT_ROOT, survey_clusters_file, survey_dirs)

SECTIONS = ("donnees", "dhs", "ehcvm", "externe", "articles", "carte")
LABELS = {
    "reference_moyenne": "Référence : moyenne",
    "reference_milieu": "Référence : urbain/rural",
    "viirs": "VIIRS seul (2 km U / 5 km R)",
    "viirs5": "VIIRS seul (5 km pour tous)",
    "cnn_imagenet": "CNN ImageNet, image entière",
    "cnn_imagenet_tiles": "CNN ImageNet, tuiles 1 km",
    "cnn_national": "CNN national, image entière",
    "cnn_national_tiles": "CNN national, tuiles 1 km",
    "cnn_imagenet+viirs": "CNN ImageNet + VIIRS",
    "cnn_imagenet_tiles+viirs": "CNN ImageNet tuiles + VIIRS",
    "cnn_national_tiles+viirs": "CNN national tuiles + VIIRS",
    "cnn_imagenet_tiles+viirs5": "CNN ImageNet tuiles + VIIRS 5 km",
}
SCHEMES = {"random": "Aléatoire", "blocks": "Blocs 100 km", "kmeans": "KMeans"}



def title(text):
    print("\n" + "=" * 78 + f"\n  {text}\n" + "=" * 78)


def subtitle(text):
    print(f"\n--- {text} " + "-" * max(0, 72 - len(text)))


def read(path):
    if not Path(path).exists():
        print(f"  (absent : {Path(path).relative_to(PROJECT_ROOT)} — lancer l'étape correspondante)")
        return None
    return pd.read_csv(path)


def fmt(value, digits=3):
    return "—" if pd.isna(value) else f"{value:.{digits}f}"


def show_data():
    title("1. DONNÉES")
    for survey in ("dhs", "ehcvm"):
        frame = read(survey_clusters_file(survey))
        if frame is None:
            continue
        corr = np.corrcoef(np.log1p(frame["viirs_2u5r_mean"].clip(lower=0)), frame["wealth_mean"])[0, 1]
        counts = frame["urban_rural"].value_counts()
        print(f"  {survey.upper():6s} {len(frame):5d} grappes (U {counts.get('U', 0)} / R {counts.get('R', 0)})"
              f" | ménages {int(frame['n_households'].sum()):6d}"
              f" | corrélation log(VIIRS)–richesse = {corr:.2f}")
    diagnostics = read(EHCVM_TABLES_DIR / "wealth_index_diagnostics.csv")
    if diagnostics is not None:
        subtitle("Indice EHCVM (ACP pondérée)")
        for _, row in diagnostics.iterrows():
            print(f"  {row['indicateur']:34s} {row['valeur']:.3f}")
    loadings = read(EHCVM_TABLES_DIR / "wealth_index_loadings.csv")
    if loadings is not None:
        subtitle("Variables les plus liées à l'indice EHCVM (contribution à l'axe 1)")
        top = pd.concat([loadings.head(6), loadings.tail(4)])
        for _, row in top.iterrows():
            print(f"  {row['variable']:18s} {row['loading']:+.3f}")


def show_ablation(survey):
    tables = survey_dirs(survey)["tables"]
    title(f"{'2' if survey == 'dhs' else '3'}. {survey.upper()} — APPORT DE CHAQUE SOURCE (R² hors échantillon)")
    metrics = read(tables / "ablation_metrics.csv")
    if metrics is None:
        return
    pivot = metrics.pivot_table(index="feature_set", columns="scheme", values="R2")
    pivot = pivot[[s for s in SCHEMES if s in pivot.columns]]
    blocks = metrics[metrics["scheme"] == "blocks"].set_index("feature_set")
    pivot = pivot.sort_values("blocks", ascending=False) if "blocks" in pivot else pivot
    print(f"\n  {'Jeu de variables':36s}" + "".join(f"{SCHEMES[s]:>14s}" for s in pivot.columns)
          + "   IC 95 % (blocs)   écart-type plis")
    for name, row in pivot.iterrows():
        ci = f"[{blocks.loc[name, 'R2_ci_low']:.2f} ; {blocks.loc[name, 'R2_ci_high']:.2f}]" if name in blocks.index else ""
        std = fmt(blocks.loc[name, "R2_fold_std"]) if name in blocks.index else ""
        print(f"  {LABELS.get(name, name):36s}" + "".join(f"{fmt(v):>14s}" for v in row) + f"   {ci:17s} {std:>8s}")

    gains = read(tables / "ablation_gain_vs_viirs.csv")
    if gains is not None:
        subtitle("Gain par rapport à VIIRS seul (règle : gain ≥ 0,02 ET IC excluant 0)")
        combos = gains[gains["feature_set"].str.contains(r"\+")]
        for scheme in [s for s in SCHEMES if s in set(combos["scheme"])]:
            print(f"  {SCHEMES[scheme]}")
            for _, row in combos[combos["scheme"] == scheme].iterrows():
                verdict = "OUI" if row["meilleur_que_viirs"] else "non"
                print(f"    {LABELS.get(row['feature_set'], row['feature_set']):34s} {row['gain_R2']:+.3f} "
                      f"[{row['gain_ci_low']:+.3f} ; {row['gain_ci_high']:+.3f}]   meilleur : {verdict}")

    subgroups = read(tables / "ablation_subgroups.csv")
    if subgroups is not None:
        subtitle("R² par milieu (blocs 100 km)")
        table = subgroups[subgroups["scheme"] == "blocks"].pivot_table(index="feature_set", columns="subgroup", values="R2")
        keep = [k for k in ["reference_milieu", "viirs", "cnn_imagenet_tiles", "cnn_imagenet_tiles+viirs",
                            "cnn_national_tiles+viirs"] if k in table.index]
        print(f"  {'Jeu de variables':36s}{'Rural':>10s}{'Urbain':>10s}")
        for name in keep:
            print(f"  {LABELS.get(name, name):36s}{fmt(table.loc[name].get('R')):>10s}{fmt(table.loc[name].get('U')):>10s}")


def show_external():
    title("4. VALIDATION EXTERNE ENTRE ENQUÊTES (entraîné sur l'une, testé sur l'autre)")
    frame = read(COMPARISON_OUTPUTS_DIR / "cross_survey_metrics.csv")
    if frame is None:
        return
    print("  R² : accord après standardisation séparée, pas calibration sur l’échelle d’origine.")
    metric = 'standardized_agreement_R2' if 'standardized_agreement_R2' in frame else 'R2'
    print(f"\n  {'Sens':16s}{'Jeu de variables':36s}{'Spearman':>10s}{'R² std.':>8s}")
    for _, row in frame.sort_values(["train", "Spearman"], ascending=[True, False]).iterrows():
        print(f"  {row['train'].upper() + ' → ' + row['test'].upper():16s}"
              f"{LABELS.get(row['feature_set'], row['feature_set']):36s}{row['Spearman']:10.3f}{row[metric]:8.3f}")


def show_articles():
    title("5. RÉPLICATION DE JEAN ET AL. (2016)")
    for survey in ("dhs", "ehcvm"):
        tables = survey_dirs(survey)["tables"]
        subtitle(f"{survey.upper()} — test placebo (images réattribuées au hasard, CNN ImageNet tuiles)")
        placebo = read(tables / "jean_placebo.csv")
        if placebo is not None:
            print(f"  {'Validation':14s}{'R² vraies images':>18s}{'R² placebo moyen':>18s}{'placebo max':>13s}{'p-valeur':>10s}")
            for _, row in placebo.iterrows():
                print(f"  {SCHEMES[row['scheme']]:14s}{row['R2_vraies_images']:18.3f}{row['R2_placebo_moyen']:18.3f}"
                      f"{row['R2_placebo_max']:13.3f}{row['p_valeur']:10.3f}")
        subtitle(f"{survey.upper()} — R² en ne gardant que les grappes les plus pauvres (blocs 100 km)")
        quantiles = read(tables / "jean_quantiles.csv")
        if quantiles is not None:
            table = quantiles[quantiles["scheme"] == "blocks"].pivot_table(
                index="quantile", columns="feature_set", values="R2")
            order = [c for c in ["viirs", "cnn_imagenet_tiles", "cnn_imagenet_tiles+viirs"] if c in table]
            print(f"  {'Grappes gardées':18s}" + "".join(f"{LABELS[c][:26]:>28s}" for c in order))
            for q, row in table[order].iterrows():
                n = int(quantiles[(quantiles["quantile"] == q)]["n_grappes"].iloc[0])
                print(f"  {'≤ ' + str(int(q * 100)) + ' % (' + str(n) + ')':18s}" + "".join(f"{fmt(v):>28s}" for v in row))


def show_map():
    title("6. CARTE NATIONALE EXPÉRIMENTALE")
    frame = read(COMMON_TABLES_DIR / "national_wealth_predictions.csv")
    if frame is None:
        return
    print(f"  Modèle          : {frame['model'].iloc[0]} (alpha = {frame['alpha'].iloc[0]:.0f})")
    print(f"  Points          : {len(frame)} (grille de 10 km)")
    print(f"  Hors quantiles 1–99 % des scores d’apprentissage : {int(frame['outside_training_range'].sum())}")
    q = frame["wealth_predicted"].quantile([0, .25, .5, .75, 1]).round(2).tolist()
    print(f"  Richesse prédite: min {q[0]} | Q1 {q[1]} | médiane {q[2]} | Q3 {q[3]} | max {q[4]}")
    subtitle("Zones prédites les plus riches (5 premiers points)")
    for _, row in frame.nlargest(5, "wealth_predicted").iterrows():
        print(f"  lat {row['latitude']:.2f}  lon {row['longitude']:.2f}  → {row['wealth_predicted']:.2f}")
    image = PROJECT_ROOT / "outputs/common/maps/carte_richesse_predite.png"
    print(f"\n  Image : {image.relative_to(PROJECT_ROOT)}   (ouvrir : open {image.relative_to(PROJECT_ROOT)})")
    print("  Sortie de modèle, pas une statistique officielle.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--section", choices=SECTIONS, nargs="+", default=list(SECTIONS))
    args = parser.parse_args(argv)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    load_config()
    actions = {"donnees": show_data, "dhs": lambda: show_ablation("dhs"),
               "ehcvm": lambda: show_ablation("ehcvm"), "externe": show_external,
               "articles": show_articles, "carte": show_map}
    for section in args.section:
        actions[section]()
    print("\nSynthèse rédigée : reports/results/RESULTATS.md\n")


if __name__ == "__main__":
    main()
