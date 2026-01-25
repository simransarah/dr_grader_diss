import torch
import torch.nn as nn

# A double convolution block used throughout the UNet encoder/decoder.
# Two 3x3 convolutions are applied, and then BatchNorm and ReLU.
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # Forward pass through the ConvBlock
        return self.conv(x)

# An attention gate weighs encoder feature maps before concatenating in the decoder.
# F_g = no. of channels in the gating signal (from decoder)
# F_l = no. of channels in the encoder feature map
# F_int = no. of intermediate channels in the attention mechanism
class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()

        # Maps gating signal to intermediate channel space
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        # Maps encoder feature to intermediate channel space
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        # Combines the two transformed inputs and produces attention coefficients
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # g = gating signal from decoder (coarser level)
        # x = encoder feature map to be attended
        g1 = self.W_g(g)   # projects gating signal
        x1 = self.W_x(x)   # projects encoder feature
        psi = self.relu(g1 + x1)  # combines and applies ReLU non-linearity
        psi = self.psi(psi)       # attention map in [0,1]
        # element-wise multiplication of attention coefficients with encoder features
        return x * psi

# Attention UNet combining UNet with attention gates
# num_channels = number of input image channels
# num_classes = number of output segmentation classes
class Attention_UNet(nn.Module):
    def __init__(self, num_channels = 3, num_classes = 3):
        super(Attention_UNet, self).__init__()
        self.num_channels = num_channels
        self.num_classes = num_classes
        

        # Encoder path = repeated ConvBlock followed by max-pooling
        self.inc = ConvBlock(num_channels, 64)   # initial convs
        self.down1 = nn.MaxPool2d(2)
        self.conv1 = ConvBlock(64, 128)
        self.down2 = nn.MaxPool2d(2)
        self.conv2 = ConvBlock(128, 256)
        self.down3 = nn.MaxPool2d(2)
        self.conv3 = ConvBlock(256, 512)
        self.down4 = nn.MaxPool2d(2)
        self.conv4 = ConvBlock(512, 1024)        # bottom-most conv block

        # Decoder path = up-convolutions, attention gates, and ConvBlocks after concatenation
        # Each up block halves channels then attends and concatenates the corresponding encoder features
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.att1 = AttentionGate(F_g=512, F_l=512, F_int=256)
        self.up_conv1 = ConvBlock(1024, 512)

        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.att2 = AttentionGate(F_g=256, F_l=256, F_int=128)
        self.up_conv2 = ConvBlock(512, 256)

        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.att3 = AttentionGate(F_g=128, F_l=128, F_int=64)
        self.up_conv3 = ConvBlock(256, 128)

        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.att4 = AttentionGate(F_g=64, F_l=64, F_int=32)
        self.up_conv4 = ConvBlock(128, 64)

        # Final 1x1 conv to map to the desired number of classes
        self.outc = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder = stores intermediate feature maps for skip connections
        x1 = self.inc(x)        # level 1 features

        d1 = self.down1(x1)
        x2 = self.conv1(d1)     # level 2 features

        d2 = self.down2(x2)
        x3 = self.conv2(d2)     # level 3 features

        d3 = self.down3(x3)
        x4 = self.conv3(d3)     # level 4 features

        d4 = self.down4(x4)
        x5 = self.conv4(d4)     # bottleneck features

        # Decoder = upsamples, applies attention on corresponding encoder feature, concatenates and convolves
        u1 = self.up1(x5)               # upsamples from bottleneck
        a1 = self.att1(g=u1, x=x4)      # attends encoder level 4 with gating u1
        u1 = torch.cat([a1, u1], dim=1) # concats the attended encoder features with upsampled decoder features
        u1 = self.up_conv1(u1)          # passes the concatenated features through ConvBlock

        u2 = self.up2(u1)
        a2 = self.att2(g=u2, x=x3)
        u2 = torch.cat([a2, u2], dim=1)
        u2 = self.up_conv2(u2)

        u3 = self.up3(u2)
        a3 = self.att3(g=u3, x=x2)
        u3 = torch.cat([a3, u3], dim=1)
        u3 = self.up_conv3(u3)

        u4 = self.up4(u3)
        a4 = self.att4(g=u4, x=x1)
        u4 = torch.cat([a4, u4], dim=1)
        u4 = self.up_conv4(u4)

        # Final segmentation logits
        logits = self.outc(u4)
        return logits