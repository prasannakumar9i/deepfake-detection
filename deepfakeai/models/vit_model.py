import torch.nn as nn
import timm


class ViTExtractor(nn.Module):
    """
    Vision Transformer Extractor
    """
    def __init__(
        self,
        backbone='vit_base_patch16_224', 
        out_dim=768
    ):
        super().__init__()
        self.model = timm.create_model(
            backbone, 
            pretrained=True, 
            num_classes=0
        )
        self.out_dim = out_dim

    def forward(self, x):
        return self.model(x)  # [B, out_dim]
