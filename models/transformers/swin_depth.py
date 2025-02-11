import torch
import torch.nn as nn
import torch.nn.functional as F

from .swin_transformer import SwinTransformer
from .swin_transformer_v2 import SwinTransformerV2

from .blocks import (
    FeatureFusionBlock,
    FeatureFusionBlock_custom,
    Interpolate,
    _make_scratch,
)

# from swin_transformer import SwinTransformer
# from swin_transformer_v2 import SwinTransformerV2
#
# from blocks import (
#     FeatureFusionBlock,
#     FeatureFusionBlock_custom,
#     Interpolate,
#     _make_scratch,
# )

def _make_fusion_block(features, use_bn):
    return FeatureFusionBlock_custom(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
    )

class SWINDepthModel(nn.Module):
    def __init__(self, backbone='large12', pretrained=None,
                 frozen_stages=-1, non_negative=True, use_bn=False, **kwargs):
        super().__init__()

        # norm_cfg = dict(type='BN', requires_grad=True)
        features = kwargs["features"] if "features" in kwargs else 256

        window_size = int(backbone[-2:])

        if backbone[:-2] == 'base' and window_size == 12:
            # swin v1
            embed_dim = 128
            depths = [2, 2, 18, 2]
            num_heads = [4, 8, 16, 32]
            in_channels = [128, 256, 512, 1024]
        elif backbone[:-2] == 'base' and window_size == 24:
            # swin v2
            window_size /= 2 ###### window = 12
            embed_dim = 128
            depths = [2, 2, 18, 2]
            num_heads = [4, 8, 16, 32]
            pretrained_window_sizes = [12, 12, 12, 6]
            in_channels = [128, 256, 512, 1024]
        elif backbone[:-2] == 'large':
            embed_dim = 192
            depths = [2, 2, 18, 2]
            num_heads = [6, 12, 24, 48]
            in_channels = [192, 384, 768, 1536]
        elif backbone[:-2] == 'tiny':
            embed_dim = 96
            depths = [2, 2, 6, 2]
            num_heads = [3, 6, 12, 24]
            in_channels = [96, 192, 384, 768]

        if backbone == 'base24':
            backbone_cfg = dict(
                img_size=384,
                embed_dim=embed_dim,
                depths=depths,
                num_heads=num_heads,
                window_size=window_size,
                ape=False,
                drop_path_rate=0.3,
                patch_norm=True,
                use_checkpoint=False,
                pretrained_window_sizes=pretrained_window_sizes,
                frozen_stages=frozen_stages,
            )
        else:
            backbone_cfg = dict(
                embed_dim=embed_dim,
                depths=depths,
                num_heads=num_heads,
                window_size=window_size,
                ape=False,
                drop_path_rate=0.3,
                patch_norm=True,
                use_checkpoint=False,
                frozen_stages=frozen_stages,
            )


        self.pretrained = SwinTransformerV2(**backbone_cfg) if backbone == 'base24' else SwinTransformer(**backbone_cfg)
        self.scratch = _make_scratch(
            in_channels, features, groups=1, expand=False
        )

        self.scratch.refinenet1 = FeatureFusionBlock(features)
        self.scratch.refinenet2 = FeatureFusionBlock(features)
        self.scratch.refinenet3 = FeatureFusionBlock(features)
        self.scratch.refinenet4 = FeatureFusionBlock(features)

        head = nn.Sequential(
            nn.Conv2d(features, features // 2, kernel_size=3, stride=1, padding=1),
            Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(features // 2, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(True) if non_negative else nn.Identity(),
            nn.Identity(),
        )

        self.scratch.output_conv = head

        self.init_weights(pretrained=pretrained)

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone and heads.
        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """
        print(f'== Load encoder backbone from: {pretrained}')
        self.pretrained.init_weights(pretrained=pretrained)


    def forward(self, x):

        feats = self.pretrained(x)
        layer_1_rn = self.scratch.layer1_rn(feats[0])
        layer_2_rn = self.scratch.layer2_rn(feats[1])
        layer_3_rn = self.scratch.layer3_rn(feats[2])
        layer_4_rn = self.scratch.layer4_rn(feats[3])

        path_4 = self.scratch.refinenet4(layer_4_rn)
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn)
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn)
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        out = self.scratch.output_conv(path_1)

        return out




if __name__ == '__main__':
    # net = SWINDepthModel(backbone="base12", pretrained='/media/hdd1/code/model_zoo/swin_transformer/swin_base_patch4_window12_384_22k.pth')
    net = SWINDepthModel(backbone="base24", pretrained='/media/hdd1/code/model_zoo/swin_transformer/swinv2_base_patch4_window12_192_22k.pth')
    # net = SWINDepthModel(backbone="base12", pretrained='/mnt/lustre/kxian/code/model_zoo/swin_transformer/swin_base_patch4_window12_384_22k.pth')
    # net = SWINDepthModel(backbone="base24", pretrained='/mnt/lustre/kxian/code/model_zoo/swin_transformer/swinv2_base_patch4_window12_192_22k.pth')
    print(net)
    inputs = torch.ones(4,3,384,512)
    out = net(inputs)
    print(out.size())
