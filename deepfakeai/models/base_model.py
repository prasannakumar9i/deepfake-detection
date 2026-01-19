import torch
import torch.nn as nn
import logging
from .classifier_model import ClassifierHead
from .cnn_model import CNNExtractor
from .lstm_model import TemporalLSTM
from .vit_model import ViTExtractor

logger = logging.getLogger(__name__)


class CNN_ViT_LSTM(nn.Module):
    """
    A hybrid deep learning model combining CNN, ViT, and LSTM for deepfake detection.

    - CNN: Local spatial features
    - ViT: Global spatial features
    - LSTM: Temporal features (sequence modeling)
    - Classifier: Final class prediction

    Input:  Tensor [B, Seq, C, H, W]
    Output: Tensor [B, num_classes]
    """
    def __init__(
        self,
        cnn_backbone='xception',                     # Default: efficient CNN
        vit_backbone='vit_tiny_patch16_224',         # Default: light ViT
        hidden_dim=256,
        num_classes=2
    ):
        super().__init__()

        # Detect CNN output dim based on backbone name
        if 'xception' in cnn_backbone:
            cnn_out_dim = 2048
        elif 'resnet18' in cnn_backbone:
            cnn_out_dim = 512
        elif 'mobilenetv2' in cnn_backbone:
            cnn_out_dim = 1280
        else:
            cnn_out_dim = 1024  # Fallback (safe default)

        # Detect ViT output dim based on backbone name
        if 'vit_tiny' in vit_backbone:
            vit_out_dim = 192
        elif 'vit_small' in vit_backbone:
            vit_out_dim = 384
        elif 'vit_base' in vit_backbone:
            vit_out_dim = 768
        elif 'vit_large' in vit_backbone:
            vit_out_dim = 1024
        elif 'vit_huge' in vit_backbone:
            vit_out_dim = 1280
        else:
            vit_out_dim = 512  # Fallback

        self.cnn = CNNExtractor(cnn_backbone, out_dim=cnn_out_dim)
        self.vit = ViTExtractor(vit_backbone, out_dim=vit_out_dim)

        # LSTM input is CNN + ViT features
        combined_feature_dim = cnn_out_dim + vit_out_dim

        self.temporal = TemporalLSTM(
            input_dim=combined_feature_dim,
            hidden_dim=hidden_dim
        )

        self.classifier = ClassifierHead(
            input_dim=hidden_dim,
            num_classes=num_classes
        )

        logger.info(f"[INFO] CNN_ViT_LSTM initialized with:")
        logger.info(f"       CNN output dim: {cnn_out_dim}")
        logger.info(f"       ViT output dim: {vit_out_dim}")
        logger.info(f"       LSTM input dim: {combined_feature_dim}")


    def forward(self, x_seq):
        """
        Forward pass for sequential image data.

        Args:
            x_seq: Tensor of shape [B, Seq, C, H, W]

        Returns:
            Output logits: Tensor of shape [B, num_classes]
        """
        B, S, C, H, W = x_seq.shape
        features = []

        for i in range(S):
            x = x_seq[:, i]  # [B, C, H, W]
            cnn_feat = self.cnn(x)  # [B, cnn_out_dim]
            vit_feat = self.vit(x)  # [B, vit_out_dim]
            feat = torch.cat((cnn_feat, vit_feat), dim=1)  # [B, cnn + vit]
            features.append(feat.unsqueeze(1))  # [B, 1, combined_dim]

        features = torch.cat(features, dim=1)  # [B, Seq, combined_dim]
        temporal_out = self.temporal(features)  # [B, hidden_dim]
        return self.classifier(temporal_out)   # [B, num_classes]
