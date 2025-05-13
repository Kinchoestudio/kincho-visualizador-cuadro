import torch
import torch.nn as nn

from .base_model import BaseModel
from .blocks import (
    FeatureFusionBlock_custom,
    Interpolate,
    _make_encoder,
    forward_beit,
    forward_swin,
    forward_levit,
    forward_vit,
)
from .backbones.levit import stem_b4_transpose
from timm.models.layers import get_act_layer


def _make_fusion_block(features, use_bn, size=None):
    return FeatureFusionBlock_custom(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
        size=size,
    )


class DPT(BaseModel):
    def __init__(
        self,
        head,
        features=256,
        backbone="vitb_rn50_384",
        readout="project",
        channels_last=False,
        use_bn=False,
        **kwargs
    ):
        super(DPT, self).__init__()

        # Si viene con prefijo "dpt_", lo quitamos para _make_encoder
        if backbone.startswith("dpt_"):
            backbone = backbone[len("dpt_"):]

        self.channels_last = channels_last

        # Hooks para cada arquitectura (AHORA con clave correcta)
        hooks = {
            "beitl16_512":      [5, 11, 17, 23],
            "beitl16_384":      [5, 11, 17, 23],
            "beitb16_384":      [2, 5, 8, 11],
            "swin2l24_384":     [1, 1, 17, 1],
            "swin2b24_384":     [1, 1, 17, 1],
            "swin2_tiny_256":   [1, 1, 5,  1],  # <-- corregido aquí
            "swinl12_384":      [1, 1, 17, 1],
            "next_vit_large_6m":[2, 6, 36, 39],
            "levit_384":        [3, 11, 21],
            "vitb_rn50_384":    [0, 1, 8,  11],
            "vitb16_384":       [2, 5, 8,  11],
            "vitl16_384":       [5, 11,17, 23],
        }[backbone]

        if "next_vit" in backbone:
            in_features = {
                "next_vit_large_6m": [96, 256, 512, 1024],
            }[backbone]
        else:
            in_features = None

        # Construye encoder + refinenets
        self.pretrained, self.scratch = _make_encoder(
            backbone,
            features,
            False,  # True = entrenar desde cero
            groups=1,
            expand=False,
            exportable=False,
            hooks=hooks,
            use_readout=readout,
            in_features=in_features,
        )

        self.number_layers = len(hooks)
        size_refinenet3 = None
        self.scratch.stem_transpose = None

        if "beit" in backbone:
            self.forward_transformer = forward_beit
        elif "swin" in backbone:
            self.forward_transformer = forward_swin
        elif "next_vit" in backbone:
            from .backbones.next_vit import forward_next_vit
            self.forward_transformer = forward_next_vit
        elif "levit" in backbone:
            self.forward_transformer = forward_levit
            size_refinenet3 = 7
            self.scratch.stem_transpose = stem_b4_transpose(
                256, 128, get_act_layer("hard_swish")
            )
        else:
            self.forward_transformer = forward_vit

        # Punto de fusión
        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn, size_refinenet3)
        if self.number_layers >= 4:
            self.scratch.refinenet4 = _make_fusion_block(features, use_bn)

        self.scratch.output_conv = head


    def forward(self, x):
        if self.channels_last:
            x = x.contiguous(memory_format=torch.channels_last)

        layers = self.forward_transformer(self.pretrained, x)
        if self.number_layers == 3:
            l1, l2, l3 = layers
        else:
            l1, l2, l3, l4 = layers

        rn1 = self.scratch.layer1_rn(l1)
        rn2 = self.scratch.layer2_rn(l2)
        rn3 = self.scratch.layer3_rn(l3)
        if self.number_layers >= 4:
            rn4 = self.scratch.layer4_rn(l4)

        if self.number_layers == 3:
            p3 = self.scratch.refinenet3(rn3, size=rn2.shape[2:])
        else:
            p4 = self.scratch.refinenet4(rn4, size=rn3.shape[2:])
            p3 = self.scratch.refinenet3(p4, rn3, size=rn2.shape[2:])
        p2 = self.scratch.refinenet2(p3, rn2, size=rn1.shape[2:])
        p1 = self.scratch.refinenet1(p2, rn1)

        if self.scratch.stem_transpose is not None:
            p1 = self.scratch.stem_transpose(p1)

        return self.scratch.output_conv(p1)


class DPTDepthModel(DPT):
    def __init__(self, path=None, non_negative=True, **kwargs):
        features = kwargs.pop("features", 256)
        head_f1  = kwargs.pop("head_features_1", features)
        head_f2  = kwargs.pop("head_features_2", 32)

        head = nn.Sequential(
            nn.Conv2d(head_f1, head_f1 // 2, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(head_f1 // 2, head_f2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(head_f2, 1,    kernel_size=1, stride=1, padding=0),
            nn.ReLU(True) if non_negative else nn.Identity(),
            nn.Identity(),
        )

        super().__init__(head, **kwargs)

        if path is not None:
            self.load(path)

    def forward(self, x):
        return super().forward(x).squeeze(1)
