# Politique de diffusion

Cette copie a été constituée à partir d’une liste de fichiers de code et de configuration. L’ancien historique Git n’est pas repris : il contient des sorties d’analyse déjà suivies par Git.

Sont exclus : données brutes ou transformées, coordonnées, identifiants de ménages ou de grappes, caractéristiques individuelles, images d’enquête, prédictions, cartes, modèles entraînés, journaux, notebooks et rapports d’origine. Les exemples des tests sont construits dans le code.

Le fichier `.gitignore` protège les ajouts ordinaires ; il ne supprime pas un fichier déjà suivi et peut être contourné par un ajout forcé. Le contrôle `python scripts/check_publication.py` vérifie aussi les chemins suivis quand un dépôt Git existe. Un contrôle automatique ne prouve pas à lui seul l’absence de toute information sensible dans le texte du code.

Ne pas transférer l’ancien dossier `.git`, ne pas pousser l’ancien dépôt pour publier cette copie. Toute ouverture du dépôt ou ajout de résultats demande une revue des fichiers réellement transmis. Aucun droit de redistribution des sources n’est accordé ici.
