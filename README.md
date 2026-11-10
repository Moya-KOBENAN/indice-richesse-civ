# Estimation spatiale de la richesse en Côte d’Ivoire

Projet de recherche en science des données : estimer un indice de richesse matérielle au niveau des grappes d’enquête à partir de Sentinel-2 et des lumières nocturnes VIIRS.

**Version de diffusion : code uniquement.** Aucune microdonnée, coordonnée, image extraite autour des enquêtes, prédiction, carte, modèle entraîné ou sortie d’analyse n’est distribuée. Les résultats empiriques ne sont pas reproductibles sans accès autorisé aux sources.

## Question de recherche et méthode

Les caractéristiques visuelles apportent-elles de l’information au-delà des lumières nocturnes pour prédire la richesse des grappes ?

- DHS 2021 : score officiel hv271 agrégé par grappe.
- EHCVM 2021–2022 : indice construit par ACP pondérée sur les actifs et le logement, puis agrégé par grappe.
- Références : milieu urbain/rural et VIIRS ; caractéristiques ResNet-50 ImageNet, régression Ridge et combinaison CNN + VIIRS.
- Évaluation : validation spatiale imbriquée par blocs et KMeans, ablation des variables, analyses par sous-groupes et validation entre enquêtes.

Les branches d’adaptation CNN sur les mêmes grappes que l’évaluation sont exploratoires. Elles ne constituent pas une validation indépendante.

## Installation et contrôles sans données

Python 3.12 est l’environnement de référence.

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
python scripts/check_publication.py
python -m pytest -m "not data"
python run_pipeline.py --list
```

Les tests ne nécessitant pas les données vérifient des propriétés logicielles et statistiques sur des exemples construits. Ils ne reproduisent pas les performances empiriques.

## Exécution avec un accès autorisé

Les sources d’enquête doivent être obtenues auprès de leurs producteurs selon leurs conditions d’accès. Aucun accès n’est fourni par ce dépôt.

1. Placer les sources dans `data/raw/`, exclu de Git.
2. Adapter les chemins et le projet Earth Engine dans `config/config.yaml`.
3. S’authentifier auprès de Google Earth Engine pour les étapes satellitaires.
4. Consulter `python run_pipeline.py --list`, puis lancer chaque étape explicitement.

```sh
python run_pipeline.py --pipeline dhs --stage 01 --dry-run
python run_pipeline.py --pipeline dhs --stage 01
```

Le pipeline n’enchaîne pas automatiquement les étapes. Les sorties locales sont exclues intégralement de Git.

## Parcours pour l’encadreur

Le [guide de relecture](docs/GUIDE_RELECTURE.md) relie les étapes aux scripts, explique les dépendances et liste les questions scientifiques restant à trancher. La note Word est transmise séparément.

## Structure

```text
src/data/           préparation des enquêtes et de l’imagerie
src/features/       extraction des caractéristiques visuelles
src/models/         apprentissage et prédiction
src/evaluation/     validation spatiale, ablation et transfert
src/exploration/    analyses descriptives locales
src/visualization/  production locale des cartes
src/utils/          fonctions partagées
config/             configuration scientifique sans identifiants personnels
scripts/            contrôle des fichiers destinés à la diffusion
tests/              tests unitaires, structure et qualité des données
```

## Limites et interprétation

L’indice d’actifs mesure une dimension de la richesse matérielle ; il ne se confond pas avec le revenu ou la consommation. Les indices DHS et EHCVM sont des constructions différentes. Le transfert entre enquêtes renseigne notamment sur les classements ; les erreurs calculées après standardisation séparée ne démontrent pas une calibration sur l’échelle d’origine.

Les conclusions dépendent du découpage spatial. Le gain d’un modèle doit être examiné avec son incertitude, par enquête et par milieu. La règle de décision enregistrée dans la configuration n’est pas présentée comme préenregistrée.

Les cartes éventuelles sont expérimentales, sans interprétation causale ni statut de statistique officielle. Le contrôle de l’étendue des scores prédits ne constitue pas une détection complète des zones hors distribution.

## Confidentialité et état de validation

Voir [la politique de diffusion](docs/CONFIDENTIALITE.md) et [le bilan de préparation](docs/VALIDATION.md). Aucun résultat numérique issu des données confidentielles n’est publié dans cette version. Toute diffusion ultérieure d’agrégats exige un examen spécifique.
