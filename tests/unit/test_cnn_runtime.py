"""Compatibilité des dépendances CNN sans téléchargement de poids."""
import numpy as np


def test_numpy_tensor_and_resnet_forward_without_pretrained_weights():
    import torch
    from torchvision.models import resnet50
    model = resnet50(weights=None).eval()
    tensor = torch.from_numpy(np.zeros((1, 3, 32, 32), dtype=np.float32))
    with torch.no_grad():
        result = model(tensor)
    assert result.shape == (1, 1000)
    assert torch.isfinite(result).all()
