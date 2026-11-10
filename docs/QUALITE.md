# Qualité et vérification

## Vérifications reproductibles

L’environnement de référence est Python 3.12. Les dépendances sont décrites dans `requirements.txt` et leurs versions de référence dans `requirements-lock.txt`.

```sh
python -m pip install -r requirements-lock.txt
python -m pip check
python -m pytest -m "not data" -q
python scripts/check_publication.py
```

Les tests sans données couvrent notamment les partitions spatiales, la sélection imbriquée des hyperparamètres, les références prédictives, les opérations de l’ACP pondérée, les ex æquo des quintiles, les clés de jointure, l’assemblage des caractéristiques et la configuration. Un test vérifie l’import des étapes sans écriture ni connexion réseau après initialisation des bibliothèques tierces. Le contrôle CNN utilise des poids non préentraînés et ne télécharge aucun modèle.

## Contrôles sur données autorisées

```sh
python -m pytest tests/data_quality -q
```

Ces tests nécessitent les fichiers privés indiqués dans le guide d’exécution. Ils vérifient les tables analytiques, les identifiants, les caractéristiques visuelles et l’intégrité des images. Les données, les résultats détaillés et les journaux restent dans l’espace de recherche privé.

## Portée

Les tests logiciels ne certifient pas les performances scientifiques. La reproduction empirique exige les sources autorisées, les paramètres utilisés et les poids des modèles correspondants. La validité de la cible, les sensibilités de l’indice et la généralisation spatiale font l’objet de diagnostics distincts décrits dans la méthodologie.
