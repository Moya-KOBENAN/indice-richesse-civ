"""Extracteurs ResNet-50 : ImageNet seul ou adapté aux lumières nocturnes.

Les deux extracteurs renvoient le vecteur de 2048 caractéristiques de la
couche de pooling global (la tête de classification est retirée).
"""
from pathlib import Path

import numpy as np

N_FEATURES = 2048


def device():
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_extractor(kind, checkpoint=None):
    """Renvoie (modèle en mode évaluation, transformation d'entrée ImageNet)."""
    import torch
    import torch.nn as nn
    from torchvision.models import ResNet50_Weights, resnet50

    weights = ResNet50_Weights.DEFAULT
    if kind == "imagenet":
        model = resnet50(weights=weights)
    elif kind == "national":
        if checkpoint is None or not Path(checkpoint).exists():
            raise FileNotFoundError(f"Poids du CNN national absents : {checkpoint}. Exécuter common/12.")
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        model = resnet50(weights=None)
        # Même tête qu'à l'étape 12, nécessaire pour charger les poids.
        model.fc = nn.Sequential(nn.Linear(N_FEATURES, 256), nn.ReLU(), nn.Dropout(0.30),
                                 nn.Linear(256, state.get("num_classes", 5)))
        model.load_state_dict(state["model_state_dict"], strict=True)
    else:
        raise ValueError(f"Extracteur inconnu : {kind} (imagenet | national)")
    model.fc = nn.Identity()
    return model.to(device()).eval(), weights.transforms()


def extract_mean_features(model, transform, tiles, batch_size=16):
    """Moyenne des vecteurs des tuiles d'une zone (agrégation de Xie et al.)."""
    import torch
    from PIL import Image

    vectors = []
    with torch.no_grad():
        for start in range(0, len(tiles), batch_size):
            batch = torch.stack([transform(Image.fromarray(t, mode="RGB"))
                                 for t in tiles[start:start + batch_size]])
            vectors.append(model(batch.to(device())).cpu().numpy())
    return np.concatenate(vectors).mean(axis=0)
