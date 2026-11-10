# Guide de relecture scientifique

## Parcours conseillé

1. Lire le README : objectif, unité grappe, cibles, méthodes et limites.
2. Consulter `config/config.yaml` et `python run_pipeline.py --list` pour identifier paramètres et étapes.
3. Examiner `src/data/dhs/01_prepare_dhs.py` et `02_prepare_gps.py` : échelle du score, agrégation, jointure et moyenne pondérée principale et diagnostic de la moyenne simple.
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

- La cible DHS est désormais explicitement pondérée par hv005 ; l’unité reste la grappe et ne constitue pas une moyenne nationale pondérée par la population.
- La correspondance des actifs s12q02 (1 = oui, 2 = non) a été confirmée dans le dictionnaire officiel. Poursuivre la revue des sauts et des choix de domaines ; `ehcvm/sensitivity` fournit les comparaisons d’ACP et ne valide pas à lui seul l’estimand.
- Déclarer l’exploration préalable : aucun jeu de test intact n’est revendiqué.
- Les commandes `dhs/spatial_audit` et `ehcvm/spatial_audit` mesurent les distances au plus proche point d’apprentissage. Examiner ces sorties privées avant de décider d’une zone tampon ; des groupes disjoints ne garantissent pas une séparation des fenêtres satellitaires.
- Ne pas assimiler le bootstrap des prédictions existantes à un réentraînement bootstrap complet ; sa validité dépend aussi des dépendances entre blocs.
- Présenter conjointement blocs et KMeans ; aucun schéma n’est universellement « plus strict » sans analyse des distances effectives.
- Examiner le décalage temporel, la validité des caractéristiques hors échantillon et l’incertitude des cartes avant tout usage opérationnel.

Ces points ne sont pas résolus par les tests logiciels. Toute correction de la cible, de l’échantillon ou du protocole exige de recalculer les résultats ; les anciens chiffres ne sont alors pas automatiquement applicables.

## Organisation Git

`main` contient la version vérifiée du code. Les corrections sont préparées dans `quality/research-review` et `quality/architecture-audit`, puis intégrées après contrôles. Une branche expérimentale est pertinente quand une méthode alternative modifie effectivement le code ; elle ne sert pas à répartir les étapes interdépendantes d’un même pipeline.

La note Word est transmise séparément et ne fait pas partie du dépôt.

## Documentation primaire consultée

- [Dictionnaire EHCVM : possession des actifs s12q02](https://microdata.worldbank.org/catalog/6273/variable/F23/V787?name=s12q02) : modalités oui/non.
- [Dictionnaire EHCVM : logement](https://microdata.worldbank.org/catalog/6273/data-dictionary/F22) : libellés des questions ; les modalités sont traitées comme catégories.

Ces liens concernent la documentation publique, pas les fichiers d’enquête confidentiels.
