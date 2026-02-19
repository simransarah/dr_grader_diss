import segmentation_models_pytorch as smp

def build_model(encoder_weights="imagenet"):
    return smp.Unet(
        encoder_name="efficientnet-b3",
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=1,
        decoder_attention_type="scse",
    )
