import torch.nn as nn
import timm


class CNNExtractor(nn.Module):
    """"
    CNN Feature Extractor
    """
    def __init__(
        self,
        backbone='resnet18', 
        out_dim=512
    ):
        super().__init__()
        self.model = timm.create_model(
            backbone,
            pretrained=True, 
            num_classes=0,
            global_pool='avg'
        )
        self.out_dim = out_dim

    def forward(self, x):
        return self.model(x)  # [B, out_dim]