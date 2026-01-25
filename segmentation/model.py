import torch 
import torch.nn as nn

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
    def __init__(self, num_channels = 3, num_classes = 3):
        super(Attention_UNet, self).__init__()
        # encoder 
        self.inc = ConvBlock(num_channels, 64)
        self.down1 = nn.MaxPool2d(2)
        self.conv1 = ConvBlock(64, 128)
        self.down2 = nn.MaxPool2d(2)
        self.conv2 = ConvBlock(128, 256)
        self.down3 = nn.MaxPool2d(2)
        self.conv3 = ConvBlock(256, 512)
        self.down4 = nn.MaxPool2d(2)
        self.conv4 = ConvBlock(512, 1024)

        # decoder with attention gates
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.att1 = AttentionGate(F_g=512, F_l=512, F_int=256)
        self.conv_up1 = ConvBlock(1024, 512)

        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.att2 = AttentionGate(F_g=256, F_l=256, F_int=128)
        self.conv_up2 = ConvBlock(512, 256)

        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.att3 = AttentionGate(F_g=128, F_l=128, F_int=64)
        self.conv_up3 = ConvBlock(256, 128)

        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.att4 = AttentionGate(F_g=64, F_l=64, F_int=32)
        self.conv_up4 = ConvBlock(128, 64)

        self.outc = nn.Conv2d(64, num_classes, kernel_size=1)

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
        