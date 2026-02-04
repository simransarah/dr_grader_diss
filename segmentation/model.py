import torch 
import torch.nn as nn

class ChannelAttention(nn.Module):

    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # Shared MLP
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Concatenate average and max pooling along the channel axis
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(x_cat)
        return self.sigmoid(out)

class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        out = x * self.ca(x)
        result = out * self.sa(out)
        return result

class CBAM_Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CBAM_Block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.cbam = CBAM(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.cbam(x) # Refine features with attention
        return x

class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class CBAM_AG_UNet(nn.Module):
    """
    The exact architecture from the paper: 
    Encoder (CBAM) + Bottleneck (CBAM) + Decoder (AG + CBAM).
    """
    def __init__(self, in_channels=1, out_channels=1, **kwargs):
        super(CBAM_AG_UNet, self).__init__()
        
        # Paper configuration: 32 -> 64 -> 128 -> 256 -> 512 (Bottleneck)
        filters = [32, 64, 128, 256, 512]

        # --- Encoder (with CBAM) ---
        self.inc = CBAM_Block(in_channels, filters[0])
        self.down1 = nn.MaxPool2d(2)
        
        self.conv1 = CBAM_Block(filters[0], filters[1])
        self.down2 = nn.MaxPool2d(2)
        
        self.conv2 = CBAM_Block(filters[1], filters[2])
        self.down3 = nn.MaxPool2d(2)
        
        self.conv3 = CBAM_Block(filters[2], filters[3])
        self.down4 = nn.MaxPool2d(2)
        
        # --- Bottleneck (with CBAM) ---
        self.bottleneck = CBAM_Block(filters[3], filters[4])

        # --- Decoder (AG + CBAM) ---
        # Up 1
        self.up1 = nn.ConvTranspose2d(filters[4], filters[3], kernel_size=2, stride=2)
        self.att1 = AttentionGate(F_g=filters[3], F_l=filters[3], F_int=filters[2])
        self.conv_up1 = CBAM_Block(filters[4], filters[3]) # 256+256 inputs -> 256 out

        # Up 2
        self.up2 = nn.ConvTranspose2d(filters[3], filters[2], kernel_size=2, stride=2)
        self.att2 = AttentionGate(F_g=filters[2], F_l=filters[2], F_int=filters[1])
        self.conv_up2 = CBAM_Block(filters[3], filters[2])

        # Up 3
        self.up3 = nn.ConvTranspose2d(filters[2], filters[1], kernel_size=2, stride=2)
        self.att3 = AttentionGate(F_g=filters[1], F_l=filters[1], F_int=filters[0])
        self.conv_up3 = CBAM_Block(filters[2], filters[1])

        # Up 4
        self.up4 = nn.ConvTranspose2d(filters[1], filters[0], kernel_size=2, stride=2)
        self.att4 = AttentionGate(F_g=filters[0], F_l=filters[0], F_int=int(filters[0]/2))
        self.conv_up4 = CBAM_Block(filters[1], filters[0])

        self.outc = nn.Conv2d(filters[0], out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.conv1(self.down1(x1))
        x3 = self.conv2(self.down2(x2))
        x4 = self.conv3(self.down3(x3))
        
        # Bottleneck
        x5 = self.bottleneck(self.down4(x4))

        # Decoder
        d5 = self.up1(x5)
        x4_att = self.att1(g=d5, x=x4) # Gate the skip connection
        d5 = torch.cat((x4_att, d5), dim=1)
        d5 = self.conv_up1(d5)

        d4 = self.up2(d5)
        x3_att = self.att2(g=d4, x=x3)
        d4 = torch.cat((x3_att, d4), dim=1)
        d4 = self.conv_up2(d4)

        d3 = self.up3(d4)
        x2_att = self.att3(g=d3, x=x2)
        d3 = torch.cat((x2_att, d3), dim=1)
        d3 = self.conv_up3(d3)

        d2 = self.up4(d3)
        x1_att = self.att4(g=d2, x=x1)
        d2 = torch.cat((x1_att, d2), dim=1)
        d2 = self.conv_up4(d2)

        logits = self.outc(d2)
        return logits
