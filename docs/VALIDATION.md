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

Aucun réentraînement, téléchargement, appel Earth Engine ou contrôle des données brutes n’a été exécuté. Les chiffres de performance et les conclusions empiriques ne sont pas certifiés par ces tests. L’installation depuis zéro avec le fichier de versions et le workflow GitHub Actions restent à vérifier sur une machine vierge. Plusieurs scripts historiques conservent des paramètres scientifiques en dur et des calculs au niveau du module ; l’exécution doit passer par les étapes explicites du pipeline.

La revue de confidentialité porte sur la copie préparée. Le dépôt d’origine contient des sorties dans son historique et ne doit pas servir de source à un push de cette version. La commande distante de lecture a renvoyé « Repository not found » ; elle ne permet pas de déterminer si le dépôt est inexistant ou privé et inaccessible.
