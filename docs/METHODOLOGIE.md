# Méthodologie

## Objectif et unité d’analyse

Estimer un indice continu de richesse matérielle au niveau des grappes d’enquête en Côte d’Ivoire et évaluer la contribution des images de jour au-delà des lumières nocturnes. L’estimand est un score patrimonial moyen de grappe ; il ne correspond ni au revenu individuel ni à une estimation causale.

## Sources et préparation

Les fichiers ménages et géographiques DHS 2021 fournissent le score officiel hv271 et les positions diffusées des grappes. Le score est remis à l’échelle et agrégé avec les poids hv005. La moyenne simple est conservée pour diagnostic.

Les modules EHCVM 2021–2022 sont rapprochés par les clés ménage. Les caractéristiques du logement sont représentées par des indicatrices catégorielles ; la possession de biens est codée à partir des modalités documentées. L’ACP utilise une standardisation et une covariance pondérées par hhweight. Le premier axe est orienté selon la possession de biens, puis agrégé par grappe avec les poids ménages.

Les indices DHS et EHCVM sont des constructions distinctes. La cible EHCVM est définie sur l’échantillon complet sans information satellitaire : l’évaluation prédictive est conditionnelle à cette cible, et ne constitue pas une validation de toute la construction de l’indice sur de nouveaux ménages.

Sentinel-2 apporte l’imagerie de jour et VIIRS les mesures de luminosité nocturne. Les images, leurs fenêtres temporelles et les voisinages sont définis dans la configuration et les scripts d’extraction. L’accès Earth Engine est requis pour ces étapes.

## Modèles et références

Les références comprennent la moyenne d’apprentissage, le milieu urbain/rural et VIIRS seul. ResNet-50 ImageNet extrait les caractéristiques visuelles, sur image entière ou par agrégation de tuiles. Ridge fournit une régression régularisée adaptée aux caractéristiques nombreuses et corrélées.

La combinaison CNN + VIIRS utilise une stratégie séquentielle : régression non pénalisée sur VIIRS, puis Ridge sur les résidus. Cette spécification est une variante particulière de combinaison ; elle n’est pas équivalente à toutes les régressions conjointes possibles.

L’adaptation du CNN aux classes VIIRS est une expérience complémentaire. Les variantes adaptées sur les grappes utilisées pour l’évaluation ont un statut exploratoire.

## Validation et incertitude

Les comparaisons reposent sur les mêmes grappes disponibles et les mêmes partitions externes. Les hyperparamètres sont sélectionnés dans des partitions internes ; imputation et standardisation des prédicteurs sont apprises dans l’apprentissage.

La validation compare des blocs réguliers et des groupes KMeans. Une séparation aléatoire sert de comparaison. Les métriques comprennent R², RMSE, MAE et corrélation de rang. L’ablation et les analyses par milieu précisent l’apport des sources. Le bootstrap rééchantillonne les groupes sur les prédictions existantes ; il suppose une indépendance suffisante entre groupes et ne couvre pas toute l’incertitude d’apprentissage.

Le transfert entre enquêtes porte notamment sur les classements. Les métriques `standardized_agreement_*` décrivent l’accord après standardisation séparée ; elles ne mesurent pas la calibration sur l’échelle d’origine. La proximité géographique entre enquêtes peut subsister.

## Diagnostics et limites

- `ehcvm/sensitivity` compare l’indice de référence, les actifs seuls, le logement seul et plusieurs seuils de regroupement des catégories rares.
- `dhs/spatial_audit` et `ehcvm/spatial_audit` mesurent les distances entre observations de test et d’apprentissage. Des groupes disjoints ne garantissent pas une séparation des fenêtres satellitaires.
- Le déplacement des coordonnées, les décalages temporels, le choix des variables de l’indice et les performances par milieu conditionnent l’interprétation.
- L’exploration et les expériences antérieures ont utilisé les données disponibles ; aucun test totalement intact n’est revendiqué. La règle de décision consignée n’est pas présentée comme préenregistrée.
- La cartographie est expérimentale. L’étendue des scores prédits ne remplace pas un diagnostic complet hors distribution ; une grille géographique ne représente pas une population pondérée.

## Parcours du code

| Étape | Emplacement |
|---|---|
| Préparation et agrégation DHS | `src/data/dhs/` |
| Préparation et indice EHCVM | `src/data/ehcvm/` |
| Extraction et matrices visuelles | `src/features/` |
| Modélisation partagée | `src/utils/modeling.py` |
| Partitions spatiales | `src/utils/spatial.py` |
| Comparaison des sources et transfert | `src/evaluation/common/` |
| Sensibilité de l’indice | `src/evaluation/ehcvm/` |

Les étapes et entrées attendues sont listées par `python run_pipeline.py --list`.

- DHS : `01 → 02 → 03 → eda → 05 → 06t → ablation`. L’étape `03` attend l’export VIIRS du script Earth Engine configuré avec une source privée autorisée.
- EHCVM : `audit → eda → 01 → 02 → 03 → 04 → 05 → 06t → ablation`.
- Adaptation nationale optionnelle : `10 → 10b → 10c → 10d → 11 → 12`, puis extraction `13t` pour les enquêtes.
- Transfert : `common/cross`, après les préparations des deux enquêtes.
- Cartographie nationale : `common/14 → 15 → 16`, avec les entrées indiquées dans le registre.

## Documentation des sources

- [DHS Program](https://dhsprogram.com/) : accès aux enquêtes et documentation des indices.
- [Catalogue EHCVM Côte d’Ivoire 2021–2022](https://microdata.worldbank.org/catalog/6273).
- [Dictionnaire EHCVM : possession des actifs](https://microdata.worldbank.org/catalog/6273/variable/F23/V787?name=s12q02).
- [Dictionnaire EHCVM : logement](https://microdata.worldbank.org/catalog/6273/data-dictionary/F22).
