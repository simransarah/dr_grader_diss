import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

# ──────────────────────────────────────────────────────────────────────────────
#  scSE — Concurrent Spatial & Channel Squeeze-and-Excitation
#  Applied exclusively in the decoder ("Specialist" component).
#
#  Design rationale:
#    • Encoder (EfficientNet-B3): frozen/pretrained general-purpose extractor.
#      Its own SE blocks already perform channel recalibration; adding more
#      attention here would be redundant and would risk disrupting ImageNet
#      priors that are valuable for retinal feature generalisation.
#    • Decoder (scSE): specialist component that learns to recalibrate BOTH
#      channels (cSE) and spatial positions (sSE) concurrently, making it
#      well-suited to recovering fine-grained point-lesion signal (MA, HE)
#      that would otherwise be washed out by upsampling.
# ──────────────────────────────────────────────────────────────────────────────

class ChannelSE(nn.Module):
    """
    Channel Squeeze-and-Excitation (cSE).
    Global average-pools to a descriptor, passes through a bottleneck MLP,
    and uses the result to rescale each channel of the input.
    """
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        # Guard against very small channel counts (e.g. filters[0]=32, reduction=16 → 2)
        bottleneck = max(channels // reduction, 4)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),        # (B, C, 1, 1)
            nn.Flatten(),                   # (B, C)
            nn.Linear(channels, bottleneck, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(bottleneck, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.fc(x).view(x.size(0), x.size(1), 1, 1)
        return x * scale


class SpatialSE(nn.Module):
    """
    Spatial Squeeze-and-Excitation (sSE).
    Projects all channels to a single spatial map via 1×1 conv, then uses
    that map to rescale each spatial position of the input.
    Particularly effective for point-lesion localisation (MA, HE).
    """
    def __init__(self, channels: int):
        super().__init__()
        self.proj = nn.Conv2d(channels, 1, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.sigmoid(self.proj(x))  # (B, 1, H, W)
        return x * scale


class scSE(nn.Module):
    """
    Concurrent scSE (Roy et al., 2018).
    Runs cSE and sSE in parallel and adds their outputs — this is the key
    difference from sequential CBAM. The concurrent design means spatial and
    channel recalibration reinforce each other rather than one gating the other,
    which has been shown to better recover small-region activations.
    """
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.cse = ChannelSE(channels, reduction)
        self.sse = SpatialSE(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Concurrent: both branches see the same input, outputs are summed
        return self.cse(x) + self.sse(x)


# ──────────────────────────────────────────────────────────────────────────────
#  Decoder building blocks
# ──────────────────────────────────────────────────────────────────────────────

class ConvBlock(nn.Module):
    """Standard double-conv block used at each decoder stage."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class AttentionGate(nn.Module):
    """
    Soft attention gate on the skip-connection feature map.
    Gates skip features using the decoder signal (g) before concatenation,
    suppressing irrelevant background activations in the skip path.
    Retained from the original architecture — works well alongside scSE.
    """
    def __init__(self, F_g: int, F_l: int, F_int: int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, bias=True),
            nn.BatchNorm2d(F_int),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, bias=True),
            nn.BatchNorm2d(F_int),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        psi = self.relu(self.W_g(g) + self.W_x(x))
        psi = self.psi(psi)
        return x * psi


class DecoderBlock(nn.Module):
    """
    One decoder stage:
      upsample → attention-gate skip → concat → ConvBlock → scSE
    """
    def __init__(
        self,
        in_channels: int,       # channels coming up from the deeper decoder stage
        skip_channels: int,     # channels from the encoder skip connection
        out_channels: int,      # desired output channels
        scse_reduction: int = 16,
    ):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
        self.att = AttentionGate(F_g=in_channels, F_l=skip_channels, F_int=in_channels // 2)
        self.conv = ConvBlock(in_channels + skip_channels, out_channels)
        self.scse = scSE(out_channels, reduction=scse_reduction)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        skip = self.att(g=x, x=skip)
        x = torch.cat([skip, x], dim=1)
        x = self.conv(x)
        return self.scse(x)


# ──────────────────────────────────────────────────────────────────────────────
#  EfficientNetB3_scSE_UNet
#
#  EfficientNet-B3 encoder skip-connection channel sizes (SMP convention):
#    stage 0 (stem):   40   channels  →  1/2  resolution
#    stage 1:          32   channels  →  1/4
#    stage 2:          48   channels  →  1/8
#    stage 3:         136   channels  →  1/16
#    stage 4 (bottleneck): 384 channels  →  1/32
#
#  Decoder output channels mirror the encoder in reverse:
#    [256, 128, 64, 32, 16]
# ──────────────────────────────────────────────────────────────────────────────

# EfficientNet-B3 skip channels per stage (from SMP encoder.out_channels[1:])
_EFF_B3_SKIPS = [40, 32, 48, 136, 384]   # shallowest → deepest
#                 s0   s1   s2   s3   bottleneck

class EfficientNetB3_scSE_UNet(nn.Module):
    """
    Specialist retinal lesion segmentation model.

    Architecture
    ────────────
    Encoder : EfficientNet-B3 (pretrained on ImageNet)
                – No custom attention added; the backbone's own
                  inverted-residual SE blocks handle channel recalibration.
    Decoder : 4× DecoderBlock, each containing:
                  ConvTranspose2d → AttentionGate (skip) → ConvBlock → scSE
                – scSE runs cSE ∥ sSE concurrently, recovering both
                  channel-wise feature salience and spatial point-lesion signal.
    Head    : 1×1 Conv → raw logit map (apply sigmoid externally or in loss).

    Parameters
    ──────────
    encoder_weights : 'imagenet' for pretrained, None to train from scratch.
    out_channels    : 1 for binary lesion segmentation.
    freeze_encoder  : if True, encoder weights are frozen for the first N epochs
                      (useful for warm-up fine-tuning with a pretrained backbone).
    """

    # Decoder output channel progression (deepest → shallowest)
    _DEC_FILTERS = [256, 128, 64, 32, 16]

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        encoder_weights: str = "imagenet",
        freeze_encoder: bool = False,
        scse_reduction: int = 16,
    ):
        super().__init__()

        # ── Encoder ──────────────────────────────────────────────────────────
        # Pull the pretrained encoder directly from SMP so we don't duplicate
        # weight-loading logic. We only use the encoder here; the decoder is
        # our custom scSE-enhanced stack.
        _dummy = smp.Unet(
            encoder_name="efficientnet-b3",
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=1,
        )
        self.encoder = _dummy.encoder
        del _dummy

        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        # ── Decoder ──────────────────────────────────────────────────────────
        # Skip channels are reversed: deepest skip feeds the first decoder block.
        # _EFF_B3_SKIPS = [40, 32, 48, 136, 384]
        # Reversed (skip order for decoder): [384, 136, 48, 32, 40]
        #   but we treat stage 4 (384) as the bottleneck, not a skip.
        #
        # Decoder blocks (4 stages):
        #   block 0: bottleneck(384) → skip s3(136), out=256
        #   block 1: 256            → skip s2(48),  out=128
        #   block 2: 128            → skip s1(32),  out=64
        #   block 3: 64             → skip s0(40),  out=32
        #
        # A final lightweight conv then goes to out_channels.

        bottleneck_ch   = _EFF_B3_SKIPS[4]   # 384
        skip_channels   = list(reversed(_EFF_B3_SKIPS[:4]))  # [136, 48, 32, 40]
        f               = self._DEC_FILTERS   # [256, 128, 64, 32, 16]

        self.dec0 = DecoderBlock(bottleneck_ch, skip_channels[0], f[0], scse_reduction)
        self.dec1 = DecoderBlock(f[0],          skip_channels[1], f[1], scse_reduction)
        self.dec2 = DecoderBlock(f[1],          skip_channels[2], f[2], scse_reduction)
        self.dec3 = DecoderBlock(f[2],          skip_channels[3], f[3], scse_reduction)

        # Final upsampling to full resolution (×2 from 1/2 scale)
        self.final_up   = nn.ConvTranspose2d(f[3], f[4], kernel_size=2, stride=2)
        self.final_conv = ConvBlock(f[4], f[4])
        self.final_scse = scSE(f[4], reduction=scse_reduction)

        # Segmentation head
        self.head = nn.Conv2d(f[4], out_channels, kernel_size=1)

        self._init_decoder_weights()

    def _init_decoder_weights(self):
        """Kaiming init for all decoder conv layers."""
        for m in [self.dec0, self.dec1, self.dec2, self.dec3,
                  self.final_up, self.final_conv, self.head]:
            for layer in m.modules():
                if isinstance(layer, nn.Conv2d):
                    nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
                elif isinstance(layer, nn.BatchNorm2d):
                    nn.init.ones_(layer.weight)
                    nn.init.zeros_(layer.bias)

    def freeze_encoder(self, freeze: bool = True):
        """Toggle encoder weight freezing at runtime (e.g. after warm-up epochs)."""
        for param in self.encoder.parameters():
            param.requires_grad = not freeze

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder — returns list of feature maps at each stage
        # SMP convention: features[0] = original input (unused), features[1..5] = stage outputs
        features = self.encoder(x)
        # features[1]: (B,  40, H/2,  W/2)   — s0 skip
        # features[2]: (B,  32, H/4,  W/4)   — s1 skip
        # features[3]: (B,  48, H/8,  W/8)   — s2 skip
        # features[4]: (B, 136, H/16, W/16)  — s3 skip
        # features[5]: (B, 384, H/32, W/32)  — bottleneck

        s0, s1, s2, s3, bottleneck = (
            features[1], features[2], features[3], features[4], features[5]
        )

        # Decoder — each block upsample → gate skip → concat → ConvBlock → scSE
        d = self.dec0(bottleneck, s3)   # (B, 256, H/16, W/16)
        d = self.dec1(d, s2)            # (B, 128, H/8,  W/8)
        d = self.dec2(d, s1)            # (B,  64, H/4,  W/4)
        d = self.dec3(d, s0)            # (B,  32, H/2,  W/2)

        # Final upsample to full resolution
        d = self.final_up(d)            # (B,  16, H,    W)
        d = self.final_conv(d)
        d = self.final_scse(d)

        return self.head(d)             # (B,   1, H,    W)  — raw logits
