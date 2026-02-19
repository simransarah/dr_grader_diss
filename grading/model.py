import torch
import torch.nn as nn
from torchvision import models

class HybridModel(nn.Module):
    def __init__(self, num_classes=5, embedding_dim=512, num_heads=8, num_layers=4):
        super(HybridModel, self).__init__()
        # CNN Backbone - EfficientNet-B0
        weights = models.EfficientNet_B0_Weights.DEFAULT
        self.cnn = models.efficientnet_b0(weights=weights)

        # remove the classification head
        self.feature_extractor = nn.Sequential(*list(self.cnn.children())[:-2])
        
        # project from EfficientNet-B0 output channels (1280) to embedding dim
        self.projection = nn.Conv2d(1280, embedding_dim, kernel_size=1)

        # CLS token — aggregates global representation for classification
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))
        
        # learnable positional encoding — 256 patch tokens + 1 CLS token
        self.positional_encoding = nn.Parameter(torch.randn(1, 257, embedding_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, num_classes)
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        features = self.projection(features)

        # flatten spatial dims and transpose to (B, num_patches, embedding_dim)
        features = features.flatten(2).transpose(1, 2)
        b = features.shape[0]
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, features), dim=1)
        
        x = x + self.positional_encoding
        x = self.transformer(x)

        # classify using CLS token output
        return self.classifier(x[:, 0])
