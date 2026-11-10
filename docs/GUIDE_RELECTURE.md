# Guide de relecture scientifique

## Parcours conseillé

1. Lire le README : objectif, unité grappe, cibles, méthodes et limites.
2. Consulter `config/config.yaml` et `python run_pipeline.py --list` pour identifier paramètres et étapes.
3. Examiner `src/data/dhs/01_prepare_dhs.py` et `02_prepare_gps.py` : échelle du score, agrégation, jointure et pondérations à justifier.
4. Examiner `src/data/ehcvm/01_build_household_table.py`, `02_build_wealth_index.py`, `03_aggregate_clusters.py` : clés, recodages, poids, ACP et agrégation.
5. Examiner `src/features/common/extract_cnn_features.py` : normalisation, découpage en tuiles et agrégation des caractéristiques.
6. Examiner `src/evaluation/common/compare_feature_sets.py`, `src/utils/modeling.py` et `src/utils/spatial.py` : échantillon commun, modèles, sélection interne et partitions externes.
7. Examiner `src/evaluation/common/cross_survey_validation.py` : transfert entre enquêtes, classement et accord après standardisation.

## Ordre des traitements locaux

Chaque étape est explicite ; les flèches indiquent des dépendances, pas une commande d’exécution globale.

- DHS : `01 → 02 → 03 → eda → 05 → 06t → ablation` ; `03` attend une extraction VIIRS préalable, réalisée via le script Earth Engine et un asset privé autorisé.
- EHCVM : `audit → eda → 01 → 02 → 03 → 04 → 05 → 06t → ablation`.
- Adaptation nationale optionnelle : `10 → 10b → 10c → 10d → 11 → 12`, puis extraction `13t` pour les enquêtes.
- Transfert entre enquêtes : `common/cross`, après préparation et extraction des deux enquêtes.
- Cartographie nationale expérimentale : extraction des caractéristiques `common/14`, puis `15 → 16`, avec les entrées requises indiquées dans le registre.

Le registre détaillé fait autorité pour les fichiers attendus. Les données et sorties restent exclues du dépôt. Les branches DHS `06b`, `06c`, `06d`, `07b` sont exploratoires ; leur adaptation sur les grappes évaluées ne constitue pas une évaluation indépendante.

## Points scientifiques encore ouverts

- Tracer et justifier la pondération de l’agrégation DHS, au lieu de l’inférer des noms de colonnes.
- Valider les modalités et sauts EHCVM à partir des questionnaires ; documenter les sensibilités de l’ACP et l’estimand visé.
- Déclarer l’exploration préalable : aucun jeu de test intact n’est revendiqué.
- Examiner le chevauchement des fenêtres satellitaires et les distances entre apprentissage et test. Des groupes disjoints ne garantissent pas une zone tampon spatiale.
- Ne pas assimiler le bootstrap des prédictions existantes à un réentraînement bootstrap complet ; sa validité dépend aussi des dépendances entre blocs.
- Présenter conjointement blocs et KMeans ; aucun schéma n’est universellement « plus strict » sans analyse des distances effectives.
- Examiner le décalage temporel, la validité des caractéristiques hors échantillon et l’incertitude des cartes avant tout usage opérationnel.

Ces points ne sont pas résolus par les tests logiciels. Toute correction de la cible, de l’échantillon ou du protocole exige de recalculer les résultats ; les anciens chiffres ne sont alors pas automatiquement applicables.

## Organisation Git

`main` contient la version vérifiée du code. Les corrections sont préparées dans `quality/research-review`, puis intégrées après contrôles. Une branche expérimentale est pertinente quand une méthode alternative modifie effectivement le code ; elle ne sert pas à répartir les étapes interdépendantes d’un même pipeline.

La note Word est transmise séparément et ne fait pas partie du dépôt.
