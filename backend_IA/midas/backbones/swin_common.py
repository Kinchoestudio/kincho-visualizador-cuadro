import torch
import torch.nn as nn
import numpy as np

from .utils import activations, forward_default, Transpose


def forward_swin(pretrained, x):
    return forward_default(pretrained, x)


def _make_swin_backbone(
        model,
        hooks=[1, 1, 17, 1],
        patch_grid=[96, 96]
):
    pretrained = nn.Module()
    pretrained.model = model

    # Activaciones mediante lambda (sin usar get_activation)
    pretrained.model.layers[0].blocks[hooks[0]].register_forward_hook(lambda m, i, o: activations.update({"1": o}))
    pretrained.model.layers[1].blocks[hooks[1]].register_forward_hook(lambda m, i, o: activations.update({"2": o}))
    pretrained.model.layers[2].blocks[hooks[2]].register_forward_hook(lambda m, i, o: activations.update({"3": o}))
    pretrained.model.layers[3].blocks[hooks[3]].register_forward_hook(lambda m, i, o: activations.update({"4": o}))

    pretrained.activations = activations

    patch_grid_size = (
        np.array(model.patch_grid) if hasattr(model, "patch_grid") else np.array(patch_grid)
    ).astype(int)

    # Procesamiento por capas
    pretrained.act_postprocess1 = nn.Sequential(
        Transpose(1, 2),
        nn.Unflatten(2, torch.Size(patch_grid_size.tolist()))
    )
    pretrained.act_postprocess2 = nn.Sequential(
        Transpose(1, 2),
        nn.Unflatten(2, torch.Size((patch_grid_size // 2).tolist()))
    )
    pretrained.act_postprocess3 = nn.Sequential(
        Transpose(1, 2),
        nn.Unflatten(2, torch.Size((patch_grid_size // 4).tolist()))
    )
    pretrained.act_postprocess4 = nn.Sequential(
        Transpose(1, 2),
        nn.Unflatten(2, torch.Size((patch_grid_size // 8).tolist()))
    )

    return pretrained
