import torch 
import torch.nn as nn
from monai.networks.nets import FlexibleUNet

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
        return self.conv(x)
    
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
    
class Attention_UNet(nn.Module):
    # in_channels=1 for new green channel only configuration
    def __init__(self, in_channels=1, out_channels=3, **kwargs):
        super(Attention_UNet, self).__init__()
        
        filters = [32, 64, 128, 256, 512]

        # encoder 
        self.inc = ConvBlock(in_channels, filters[0])
        self.down1 = nn.MaxPool2d(2)
        self.conv1 = ConvBlock(filters[0], filters[1])
        self.down2 = nn.MaxPool2d(2)
        self.conv2 = ConvBlock(filters[1], filters[2])
        self.down3 = nn.MaxPool2d(2)
        self.conv3 = ConvBlock(filters[2], filters[3])
        self.down4 = nn.MaxPool2d(2)
        self.conv4 = ConvBlock(filters[3], filters[4])

        # decoder with attention gates
        self.up1 = nn.ConvTranspose2d(filters[4], filters[3], kernel_size=2, stride=2)
        self.att1 = AttentionGate(F_g=filters[3], F_l=filters[3], F_int=filters[2])
        self.conv_up1 = ConvBlock(filters[4], filters[3])

        self.up2 = nn.ConvTranspose2d(filters[3], filters[2], kernel_size=2, stride=2)
        self.att2 = AttentionGate(F_g=filters[2], F_l=filters[2], F_int=filters[1])
        self.conv_up2 = ConvBlock(filters[3], filters[2])

        self.up3 = nn.ConvTranspose2d(filters[2], filters[1], kernel_size=2, stride=2)
        self.att3 = AttentionGate(F_g=filters[1], F_l=filters[1], F_int=filters[0])
        self.conv_up3 = ConvBlock(filters[2], filters[1])

        self.up4 = nn.ConvTranspose2d(filters[1], filters[0], kernel_size=2, stride=2)
        self.att4 = AttentionGate(F_g=filters[0], F_l=filters[0], F_int=int(filters[0]/2))
        self.conv_up4 = ConvBlock(filters[1], filters[0])

        self.outc = nn.Conv2d(filters[0], out_channels, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.conv1(self.down1(x1))
        x3 = self.conv2(self.down2(x2))
        x4 = self.conv3(self.down3(x3))
        x5 = self.conv4(self.down4(x4))

        # decoder 
        d5 = self.up1(x5)
        x4 = self.att1(g=d5, x=x4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.conv_up1(d5)

        d4 = self.up2(d5)
        x3 = self.att2(g=d4, x=x3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.conv_up2(d4)

        d3 = self.up3(d4)
        x2 = self.att3(g=d3, x=x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.conv_up3(d3)

        d2 = self.up4(d3)
        x1 = self.att4(g=d2, x=x1)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.conv_up4(d2)

        logits = self.outc(d2)
        return logits

class Transfer_UNet(nn.Module):
    def __init__(self, num_classes=3, backbone="efficientnet-b3", pretrained=True):
        super(Transfer_UNet, self).__init__()
        
        self.model = FlexibleUNet(
            in_channels=3,
            out_channels=num_classes,
            backbone=backbone,
            pretrained=pretrained,
            spatial_dims=2,
            is_pad=False
        )

    def forward(self, x):
        if x.dtype != torch.float32:
            x = x.float()
        return self.model(x)