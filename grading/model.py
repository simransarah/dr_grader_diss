import torch
import torch.nn as nn
from torchvision import models

class HybridModel(nn.Module):
    def __init__(self, num_class=5, embedding_dim=512, num_heads = 8, num_layers = 4, dropout=0.1):
        super(HybridModel, self).__init__()
        
        # CNN Backbone - EfficientNet-B0
        weights = models.EfficientNet_B0_Weights.DEFAULT
        self.cnn = models.efficientnet_b0(weights=weights)

        original_conv = self.cnn.features[0][0]
        self.cnn.features[0][0] = nn.Conv2d(
           in_channels=4,
              out_channels=original_conv.out_channels,
                kernel_size=original_conv.kernel_size,
                stride=original_conv.stride,
                padding=original_conv.padding,
                bias=False
        )

        with torch.no_grad():
            self.cnn.features[0][0].weight[:, :3, :, :] = original_conv.weight
            self.cnn.features[0][0].weight[:, 3:, :, :] = torch.mean(original_conv.weight, dim=1, keepdim=True)

        self.feature_extractor = nn.Sequential(*list(self.cnn.children())[:-2])
        self.projection = nn.Conv2d(1280, embedding_dim, kernel_size=1)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))
        self.positional_encoding = nn.Parameter(torch.randn(1,257, embedding_dim))

        
        encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = nn.Sequential(
            nn.LayerNorm(embedding_dim),
            nn.Linear(embedding_dim, num_class)
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        features = self.projection(features)

        features = features.flatten(2).transpose(1, 2)
        b = features.shape[0]
        cls_tokens = self.cls_token.expand(b, -1, -1)
        x = torch.cat((cls_tokens, features), dim=1)
        
    
        x = x + self.positional_encoding 

        x = self.transformer(x) 

        return self.classifier(x[:, 0])  # use cls token output for classification