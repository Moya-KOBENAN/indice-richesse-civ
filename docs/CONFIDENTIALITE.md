# Données et confidentialité

Le dépôt distribue le code, sa configuration publique, les tests et la documentation méthodologique. L’accès aux sources doit être obtenu séparément auprès de leurs producteurs, selon leurs conditions d’utilisation.

## Éléments exclus

Les microdonnées, identifiants, coordonnées, tables transformées, images extraites autour des grappes, caractéristiques individuelles, prédictions, cartes, modèles entraînés, journaux et documents de travail confidentiels ne sont pas distribués. Les exemples des tests sont synthétiques.

Les données et les sorties peuvent rester dans des dossiers externes via `RICHESSE_DATA_DIR` et `RICHESSE_OUTPUTS_DIR`. Les chemins privés et paramètres d’accès sont définis dans une configuration externe ; voir le [guide d’exécution](GUIDE_EXECUTION.md).

## Contrôle de diffusion

```sh
python scripts/check_publication.py
```

Le contrôle vérifie les chemins autorisés, le contenu des fichiers de travail et de l’index Git, ainsi que certains motifs sensibles. Le fichier `.gitignore` exclut les données et sorties des ajouts ordinaires. Ces protections complètent la revue humaine des fichiers destinés à la diffusion ; elles ne constituent pas une preuve universelle d’absence d’information sensible.

Le dépôt n’accorde aucun droit de redistribution des données sources.
