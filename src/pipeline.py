"""Registre des étapes et exécution explicite, sans calcul à l'import.

Chaque étape = un script autonome dans une couche (data, features, models,
evaluation, visualization, exploration). Le registre documente l'ordre, les
entrées attendues et permet de lancer une étape depuis n'importe quel
répertoire. Aucun enchaînement automatique : chaque étape est lancée,
contrôlée, puis la suivante.
"""
from dataclasses import dataclass
from pathlib import Path
import argparse
import subprocess
import sys

from src.utils.paths import (COMMON_MODELS_DIR, COMMON_PROCESSED_DIR, COMMON_TABLES_DIR,
                             DHS_INTERIM_DIR, DHS_MODELS_DIR, DHS_PROCESSED_DIR, DHS_TABLES_DIR,
                             EHCVM_INTERIM_DIR, PROJECT_ROOT, RAW_DIR, survey_clusters_file)


@dataclass(frozen=True)
class Stage:
    script: str
    description: str
    inputs: tuple[Path, ...] = ()
    args: tuple[str, ...] = ()
    # Le script sait vérifier ses entrées lui-même avec --dry-run.
    native_dry_run: bool = False
    unavailable: str = ""

    @property
    def path(self):
        return PROJECT_ROOT / "src" / self.script


DHS_MAIN = survey_clusters_file("dhs")
EHCVM_MAIN = survey_clusters_file("ehcvm")
FEATURES = DHS_PROCESSED_DIR / "cnn_features_resnet50.csv"
SATELLITE = RAW_DIR / "satellite"
GRID = COMMON_PROCESSED_DIR / "cote_ivoire_prediction_grid.geojson"
NATIONAL_DATA = COMMON_PROCESSED_DIR / "cote_ivoire_national_grid_viirs_diagnostic.csv"
NATIONAL_CNN = COMMON_MODELS_DIR / "resnet50_national_viirs_finetuned.pt"
EXTRACT = "features/common/extract_cnn_features.py"
ABLATION = "evaluation/common/compare_feature_sets.py"
JEAN = "evaluation/common/replicate_jean_2016.py"

PIPELINES = {
    "dhs": {
        "01": Stage("data/dhs/01_prepare_dhs.py", "Indice de richesse par grappe (fichier HR)"),
        "02": Stage("data/dhs/02_prepare_gps.py", "GPS des grappes et fusion", (DHS_INTERIM_DIR / "dhs_clusters_wealth.csv",)),
        "03": Stage("data/dhs/03_merge_viirs.py", "Fusion richesse × VIIRS", (DHS_INTERIM_DIR / "DHS_CI_2021_clusters_richesse_GPS.csv", RAW_DIR / "DHS_CI_2021_VIIRS_clusters.csv")),
        "eda": Stage("exploration/dhs/eda.py", "Analyse exploratoire DHS", (DHS_MAIN,), native_dry_run=True),
        "04": Stage("models/dhs/04_viirs_baseline.py", "Référence VIIRS (choix du tampon)", (DHS_MAIN,)),
        "05": Stage("data/dhs/05_prepare_satellite.py", "Téléchargement Sentinel-2 (Earth Engine)", (DHS_MAIN,)),
        "06": Stage("features/dhs/06_extract_cnn_features.py", "Caractéristiques ImageNet, image entière", (SATELLITE,)),
        "06t": Stage(EXTRACT, "Caractéristiques ImageNet, tuiles de 1 km", (SATELLITE,), ("--zone", "dhs", "--extractor", "imagenet", "--tiles"), True),
        "06b": Stage("models/dhs/06b_train_cnn_viirs_exploratory.py", "Adaptation VIIRS sur DHS — exploratoire", (DHS_MAIN, SATELLITE)),
        "06c": Stage("models/dhs/06c_finetune_cnn_viirs_exploratory.py", "Ajustement CNN DHS — exploratoire", (DHS_MAIN, SATELLITE, DHS_MODELS_DIR / "resnet50_viirs_adapted.pt")),
        "06d": Stage("features/dhs/06d_extract_finetuned_features_exploratory.py", "Caractéristiques adaptées — exploratoire", (SATELLITE, DHS_MODELS_DIR / "resnet50_viirs_finetuned.pt")),
        "07": Stage("models/dhs/07_ridge_model.py", "Comparaison Ridge ImageNet/VIIRS (historique)", (DHS_MAIN, FEATURES)),
        "07b": Stage("models/dhs/07b_ridge_viirs_adapted_exploratory.py", "Ridge adapté — exploratoire", (DHS_MAIN, FEATURES, DHS_PROCESSED_DIR / "cnn_features_resnet50_viirs.csv")),
        "08": Stage("evaluation/dhs/08_validation.py", "Validation spatiale KMeans (historique)", (DHS_MAIN, FEATURES)),
        "08b": Stage("evaluation/dhs/08b_spatial_robustness.py", "Robustesse par blocs réguliers (historique)", (DHS_MAIN, FEATURES)),
        "09": Stage("visualization/dhs/09_mapping.py", "Cartes diagnostiques DHS", (DHS_TABLES_DIR / "spatial_block_robustness_predictions.csv",)),
        "09b": Stage("visualization/dhs/09b_publication_maps.py", "Cartes de publication", (DHS_TABLES_DIR / "spatial_block_robustness_predictions.csv", RAW_DIR / "cote_ivoire_boundary.geojson")),
        "13": Stage("features/dhs/13_extract_national_adapted_features.py", "CNN national sur DHS, image entière", (SATELLITE, NATIONAL_CNN)),
        "13t": Stage(EXTRACT, "CNN national sur DHS, tuiles de 1 km", (SATELLITE, NATIONAL_CNN), ("--zone", "dhs", "--extractor", "national", "--tiles"), True),
        "ablation": Stage(ABLATION, "Tableau d'ablation des sources (critère principal)", (DHS_MAIN,), ("--survey", "dhs"), True),
        "jean": Stage(JEAN, "Réplication Jean et al. 2016 : placebo et plus pauvres", (DHS_MAIN,), ("--survey", "dhs"), True),
    },
    "ehcvm": {
        "audit": Stage("data/ehcvm/audit_archive.py", "Audit de l'archive EHCVM 2021–2022", native_dry_run=True),
        "eda": Stage("exploration/ehcvm/eda.py", "Analyse exploratoire EHCVM", native_dry_run=True),
        "01": Stage("data/ehcvm/01_build_household_table.py", "Table ménage : welfare + logement + actifs + GPS"),
        "02": Stage("data/ehcvm/02_build_wealth_index.py", "Indice de richesse par ACP pondérée", (EHCVM_INTERIM_DIR / "ehcvm_households.csv",)),
        "03": Stage("data/ehcvm/03_aggregate_clusters.py", "Agrégation par grappe", (EHCVM_INTERIM_DIR / "ehcvm_households_wealth.csv",)),
        "04": Stage("data/ehcvm/04_extract_viirs.py", "VIIRS autour des grappes (Earth Engine)", (EHCVM_INTERIM_DIR / "ehcvm_clusters_wealth_gps.csv",), native_dry_run=True),
        "05": Stage("data/ehcvm/05_download_sentinel2.py", "Images Sentinel-2 des grappes (Earth Engine)", (EHCVM_MAIN,), native_dry_run=True),
        "06t": Stage(EXTRACT, "Caractéristiques ImageNet, tuiles de 1 km", (EHCVM_MAIN,), ("--zone", "ehcvm", "--extractor", "imagenet", "--tiles"), True),
        "13t": Stage(EXTRACT, "CNN national, tuiles de 1 km", (EHCVM_MAIN, NATIONAL_CNN), ("--zone", "ehcvm", "--extractor", "national", "--tiles"), True),
        "ablation": Stage(ABLATION, "Tableau d'ablation des sources", (EHCVM_MAIN,), ("--survey", "ehcvm"), True),
        "jean": Stage(JEAN, "Réplication Jean et al. 2016 : placebo et plus pauvres", (EHCVM_MAIN,), ("--survey", "ehcvm"), True),
    },
    "common": {
        "10": Stage("data/common/10_create_national_grid.py", "Grille nationale", (RAW_DIR / "cote_ivoire_boundary.geojson",)),
        "10b": Stage("data/common/10b_prepare_national_viirs.py", "VIIRS national (Earth Engine)", (GRID,)),
        "10c": Stage("data/common/10c_analyse_national_viirs.py", "Classes VIIRS nationales", (COMMON_PROCESSED_DIR / "cote_ivoire_national_grid_viirs.csv",)),
        "10d": Stage("data/common/10d_extract_grid_viirs_buffers.py", "VIIRS grille, tampon 5 km comme les grappes (Earth Engine)", (NATIONAL_DATA,), native_dry_run=True),
        "11": Stage("data/common/11_extract_national_satellite.py", "Images nationales Sentinel-2 (Earth Engine)", (GRID,)),
        "12": Stage("models/common/12_train_national_viirs_classifier.py", "Adaptation nationale ResNet-50", (NATIONAL_DATA, RAW_DIR / "satellite_national")),
        "14": Stage(EXTRACT, "Grille : caractéristiques ImageNet, tuiles", (RAW_DIR / "satellite_national",), ("--zone", "grid", "--extractor", "imagenet", "--tiles"), True),
        "14n": Stage(EXTRACT, "Grille : CNN national, tuiles", (RAW_DIR / "satellite_national", NATIONAL_CNN), ("--zone", "grid", "--extractor", "national", "--tiles"), True),
        "cross": Stage("evaluation/common/cross_survey_validation.py", "Validation externe DHS ↔ EHCVM", (DHS_MAIN, EHCVM_MAIN)),
        "15": Stage("models/common/predict_national_grid.py", "Prédiction sur la grille nationale", (COMMON_PROCESSED_DIR / "cote_ivoire_national_grid_viirs_buffers.csv",), native_dry_run=True),
        "16": Stage("visualization/common/map_national_predictions.py", "Carte nationale expérimentale", (COMMON_TABLES_DIR / "national_wealth_predictions.csv",)),
    },
}


def get_stage(pipeline, stage):
    try:
        return PIPELINES[pipeline][stage]
    except KeyError as exc:
        raise ValueError(f"Étape inconnue : {pipeline}/{stage}. Consulter --list.") from exc


def run_stage(pipeline, stage, *, dry_run=False, extra_args=()):
    selected = get_stage(pipeline, stage)
    if selected.unavailable:
        raise ValueError(selected.unavailable)
    if not selected.path.is_file():
        raise FileNotFoundError(f"Script absent : {selected.path}")
    missing = [str(path.relative_to(PROJECT_ROOT)) for path in selected.inputs if not path.exists()]
    if missing:
        raise FileNotFoundError("Entrées manquantes : " + ", ".join(missing))
    command = [sys.executable, str(selected.path), *selected.args, *extra_args]
    if dry_run and selected.native_dry_run:
        # Le script contrôle lui-même ses entrées, sans calcul ni écriture.
        command.append("--dry-run")
    elif dry_run:
        print(f"{pipeline}/{stage} : {selected.description}")
        print("Script :", selected.path)
        print("Vérification des chemins terminée ; aucune étape exécutée.")
        return command
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    return command


def main(argv=None):
    parser = argparse.ArgumentParser(description="Pipelines de richesse : DHS, EHCVM et commun (national)")
    parser.add_argument("--pipeline", choices=PIPELINES, default="dhs")
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--list", action="store_true", help="Lister les étapes de toutes les branches")
    actions.add_argument("--stage", help="Une étape précise ; voir --list")
    actions.add_argument("--tests", action="store_true")
    actions.add_argument("--resultats", action="store_true", help="Afficher tous les résultats (sans recalcul)")
    parser.add_argument("--section", nargs="+", help="Avec --resultats : donnees, dhs, ehcvm, externe, articles, carte")
    parser.add_argument("--dry-run", action="store_true", help="Vérifier les entrées sans exécuter l'étape")
    parser.add_argument("--archive", type=Path, help="Archive ZIP pour ehcvm/audit")
    parser.add_argument("--output-dir", type=Path, help="Répertoire des rapports pour ehcvm/audit")
    args = parser.parse_args(argv)
    if args.dry_run and not args.stage:
        parser.error("--dry-run nécessite --stage.")
    if (args.archive or args.output_dir) and (args.pipeline, args.stage) not in {("ehcvm", "audit"), ("ehcvm", "eda")}:
        parser.error("--archive et --output-dir sont réservés à ehcvm/audit et ehcvm/eda.")
    if args.list:
        for scope, stages in PIPELINES.items():
            print(f"\n{scope.upper()}")
            for code, stage in stages.items():
                status = " [NON IMPLÉMENTÉE]" if stage.unavailable else ""
                print(f"  {code:9} {stage.description}{status}\n            src/{stage.script} {' '.join(stage.args)}")
        return
    if args.resultats:
        extra = ["--section", *args.section] if args.section else []
        subprocess.run([sys.executable, str(PROJECT_ROOT / "src/visualization/common/print_results.py"), *extra],
                       cwd=PROJECT_ROOT, check=True)
        return
    if args.tests:
        subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=PROJECT_ROOT, check=True)
        return
    if args.stage:
        extra = []
        for name in ("archive", "output_dir"):
            value = getattr(args, name)
            if value:
                extra.extend(["--" + name.replace("_", "-"), str(value.expanduser().resolve())])
        try:
            run_stage(args.pipeline, args.stage, dry_run=args.dry_run, extra_args=extra)
        except (ValueError, FileNotFoundError) as exc:
            parser.error(str(exc))
        except subprocess.CalledProcessError as exc:
            raise SystemExit(exc.returncode) from exc
        return
    parser.print_help()
