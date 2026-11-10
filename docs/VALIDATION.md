# Bilan de préparation — 10 novembre 2026

## Vérifications exécutées localement

- Python 3.12, environnement Conda `richesse` existant.
- `python -m pytest -m "not data" -q` : 29 tests réussis, 24 tests de données exclus. Avertissement de détection des cœurs physiques de joblib, sans échec.
- `python run_pipeline.py --list` : réussi sans données.
- `python scripts/check_publication.py` : réussi pour la liste de fichiers de diffusion.
- Recherche ciblée dans le code et la configuration : retrait des chemins personnels et des identifiants de projet Earth Engine repérés.

## Corrections apportées à la copie de diffusion

- Retrait de 539 grappes (coordonnées et scores intégrés au script JavaScript) ; remplacement par une référence à un asset privé à configurer.
- Inclusion des scripts `src/data/`, omis du suivi Git d’origine par une règle `data/` trop large.
- Exclusion intégrale des données et des sorties ; aucun historique Git d’origine copié.
- Chemins de configuration génériques et projet Earth Engine lu depuis la configuration dans trois scripts historiques.
- Tests CLI corrigés : vérification explicite du refus d’exécuter une étape lorsque les entrées sont absentes.
- Documentation des limites scientifiques et ajout d’un contrôle automatique des fichiers publiables.

## Limites de cette validation

Aucun réentraînement, téléchargement, appel Earth Engine ou contrôle des données brutes n’a été exécuté. Les chiffres de performance et les conclusions empiriques ne sont pas certifiés par ces tests. À ce stade initial, l’installation depuis zéro restait à vérifier ; elle a ensuite été contrôlée pendant l’audit d’architecture décrit ci-dessous. Plusieurs scripts historiques conservent des paramètres scientifiques en dur et des calculs au niveau du module ; l’exécution doit passer par les étapes explicites du pipeline.

La revue de confidentialité porte sur la copie préparée. Le dépôt d’origine contient des sorties dans son historique et ne doit pas servir de source à un push de cette version. La commande distante de lecture a renvoyé « Repository not found » ; elle ne permet pas de déterminer si le dépôt est inexistant ou privé et inaccessible.

## Revue complémentaire du code

- Contrôle des partitions : indices valides et uniques, ensembles d’apprentissage et de test non vides, groupes disjoints, exactement une prédiction de test par observation. Ces contrôles sont appliqués avant la Ridge imbriquée.
- Standardisation entre enquêtes : convention de variance homogène, rejet des cibles et prédictions constantes ou non finies ; colonnes d’erreur renommées `standardized_agreement_*` pour ne pas les confondre avec des erreurs sur l’échelle d’origine. Les anciennes sorties ne sont pas recalculées.
- Comparaisons entre enquêtes exécutées dans un ordre déterministe ; retrait d’une affirmation injustifiée de généralisation maximale.
- Contrôle de diffusion étendu au contenu de l’index Git et du dossier de travail, avec rejet de motifs sensibles et des liens symboliques. Il s’agit d’heuristiques, pas d’une preuve universelle d’absence de données confidentielles.
- Tests après corrections : 43 réussis, 24 nécessitant les données exclus. Aucun réentraînement des enquêtes n’est effectué.
- Guide de relecture ajouté pour l’encadreur ; les limites nécessitant des données sont explicites.

Les protections sont plus exigeantes, mais le travail ne peut être qualifié de scientifiquement irréprochable sur cette seule base. Les points ouverts du guide doivent être traités avant de figer les conclusions empiriques.

## Audit d’architecture et installation propre

- Installation réellement exécutée dans un nouvel environnement Python 3.12 : l’ancien gel Conda échouait sur `torch==2.5.1.post7`. Il a été remplacé par un gel pip vérifié, sans dépendances macOS inutiles ; `pip check` réussit.
- Toutes les étapes exposent `main()` ; les corps des 18 scripts historiques ont été déplacés sans modification de leur AST. Le test d’import interdit écritures et connexions réseau après initialisation des bibliothèques tierces.
- Assemblage des matrices extrait dans `src/features/design.py` ; suppression de la dépendance des modèles envers les points d’entrée d’évaluation. Tri numérique des colonnes de caractéristiques et contrôles des cibles, coordonnées et modalités.
- Configuration privée externe et racines de données/sorties externes prises en charge ; l’audit de l’archive utilise désormais la même configuration que les autres étapes.
- Cible DHS explicitement pondérée ; moyenne simple conservée. L’ACP refuse les poids invalides et les matrices constantes ou non finies ; les valeurs manquantes des actifs ne sont plus assimilées à une absence pour calculer leur prévalence. Les quintiles préservent les ex æquo.
- Diagnostic de sensibilité ACP et audit des distances spatiales ajoutés au registre ; leurs résultats restent dans l’espace privé. Ces diagnostics ne sont pas des preuves de validité confirmatoire.
- Contrôles locaux de cohérence des sources effectués sans publier leurs valeurs ; reproduction numérique de la référence ACP vérifiée localement. Les fichiers sources n’ont pas été modifiés.
- Tests de qualité des données : 24 réussis sur les sources locales externes au dépôt.

L’importabilité des scripts ne garantit pas l’équivalence de tous leurs effets à l’exécution complète : le réentraînement CNN, les téléchargements Earth Engine et une nouvelle validation prédictive exhaustive n’ont pas été relancés. Les scripts historiques restent longs ; les classes de leurs DataLoader ne doivent pas être utilisées avec des workers multiples sans extraction au niveau du module. Voir l’audit d’architecture pour les limites de traçabilité.

Bilan final local de cet audit : **66 tests sans données réussis**, **24 tests de qualité des données réussis** dans une exécution distincte ; contrôle de confidentialité et `git diff --check` réussis.
