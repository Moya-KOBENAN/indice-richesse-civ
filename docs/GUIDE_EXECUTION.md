# Exécution et séparation des données

## Installer

Avec Python 3.12, depuis la racine du dépôt :

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python -m pip check
python -m pytest -m "not data" -q
python run_pipeline.py --list
```

Le fichier de versions provient d’un environnement pip propre testé sur macOS ARM64. Les dépendances propres à la plateforme sont résolues par pip. Les poids ImageNet ne sont pas téléchargés par les tests sans données.

## Garder les sources à l’extérieur du dépôt

Définir les variables suivantes avec ses propres chemins autorisés :

```sh
export RICHESSE_DATA_DIR=/chemin/prive/data
export RICHESSE_OUTPUTS_DIR=/chemin/prive/resultats
export RICHESSE_CONFIG_FILE=/chemin/prive/config.local.yaml
export EE_PROJECT_ID=mon-projet-autorise
```

Exemple de contenu du fichier de surcharge privé :

```yaml
dhs:
  raw_dir: /chemin/prive/sources_dhs
ehcvm:
  archive: /chemin/prive/archive_autorisee.zip
```

Les chemins relatifs de configuration se rapportent à la racine du dépôt. Les sous-dossiers attendus sous le dossier de données externe sont `raw/`, `interim/` et `processed/`. Ne pas placer d’identifiants personnels dans la configuration publique.

## Contrôler et lancer

```sh
python run_pipeline.py --pipeline dhs --stage 01 --dry-run
python run_pipeline.py --pipeline dhs --stage 01
python run_pipeline.py --pipeline ehcvm --stage sensitivity --dry-run
python run_pipeline.py --pipeline ehcvm --stage sensitivity
python run_pipeline.py --pipeline dhs --stage spatial_audit
python run_pipeline.py --pipeline ehcvm --stage spatial_audit
python -m pytest tests/data_quality -q
```

Les tests de qualité nécessitent les fichiers locaux. Le diagnostic `ehcvm/sensitivity` compare la référence ACP, les actifs seuls, le logement seul et deux seuils de regroupement des modalités rares ; il produit uniquement une table locale de sensibilité. Il ne remplace pas la validation des modèles prédictifs ni celle des choix de questionnaire.

L’ordre détaillé et les dépendances sont dans le [document méthodologique](METHODOLOGIE.md). Les étapes Earth Engine exigent un accès autorisé et une authentification ; les appels réseau se font uniquement lors de leur lancement explicite.

## Avant un envoi de code

```sh
python scripts/check_publication.py
```

Ne pas ajouter de sorties, de données ou de documents confidentiels, même avec `git add -f`. Ne pas reprendre l’ancien historique du projet d’origine. La note Word reste transmise séparément.
