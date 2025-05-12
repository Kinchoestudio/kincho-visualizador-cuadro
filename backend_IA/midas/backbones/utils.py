import torch
import torch.nn as nn
import numpy as np
from functools import partial


# Función de activación sencilla
def get_activation(name):
    if name == "relu":
        return nn.ReLU(inplace=True)
    elif name == "relu6":
        return nn.ReLU6(inplace=True)
    elif name == "leaky_relu":
        return nn.LeakyReLU(inplace=True)
    elif name == "selu":
        return nn.SELU(inplace=True)
    elif name == "gelu":
        return nn.GELU()
    else:
        raise ValueError(f"Unsupported activation function: {name}")


# Diccionario de activaciones simuladas
activations = {}

# Forward por defecto
def forward_default(model, x):
    return model(x)

# Adaptación para modelos como Swin
def forward_adapted_unflatten(model, x, method_name="forward_features"):
    x = getattr(model, method_name)(x)
    return x.view(x.size(0), -1)

# Método dummy por ahora
def make_backbone_default(model, *args, **kwargs):
    return model

# Clase para transponer tensores (usada en act_postprocessX)
class Transpose(nn.Module):
    def __init__(self, dim1, dim2):
        super().__init__()
        self.dim1 = dim1
        self.dim2 = dim2

    def forward(self, x):
        return x.transpose(self.dim1, self.dim2)


def _make_swin_backbone(model, hooks=[1, 1, 17, 1], patch_grid=[96, 96]):
    pretrained = nn.Module()

    pretrained.model = model
    pretrained.model.layers[0].blocks[hooks[0]].register_forward_hook(get_activation("relu"))
    pretrained.model.layers[1].blocks[hooks[1]].register_forward_hook(get_activation("relu"))
    pretrained.model.layers[2].blocks[hooks[2]].register_forward_hook(get_activation("relu"))
    pretrained.model.layers[3].blocks[hooks[3]].register_forward_hook(get_activation("relu"))

    pretrained.activations = activations

    patch_grid_size = np.array(patch_grid, dtype=int)

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


# 🔧 Añadido para compatibilidad con vit.py
def get_readout_oper(vit_features, features, use_readout, start_index=1):
    readout_oper = []

    for i in range(4):
        if use_readout == "ignore":
            op = lambda x: x[:, start_index:]
        elif use_readout == "add":
            op = lambda x: x[:, start_index:] + x[:, 0].unsqueeze(1)
        elif use_readout == "project":
            op = nn.Sequential(
                Transpose(1, 2),
                nn.Conv1d(vit_features, features[i], 1),
                Transpose(1, 2),
            )
        else:
            raise ValueError(f"Unknown readout operation: {use_readout}")

        readout_oper.append(op)

    return readout_oper
