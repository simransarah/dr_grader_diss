import os 
import cv2 
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
from monai.data import DataLoader, Dataset, PILReader
from monai.metrics import DiceMetric
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Resized, RandRotate90d, RandFlipd, MapTransform
)

from segmentation.config_macro import MacroConfig
from segmentation.dataset import get_monai_data_dicts
from segmentation.model import Transfer_UNet
from segmentation.loss import HybridLoss

class LoadMacroLesionsd(MapTransform):
    """Loads only HE and EX masks."""
    def __init__(self, keys, allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)
        self.reader = PILReader()

    def __call__(self, data):
        d = dict(data)
        spatial = d['image'].shape[1:] 
        lesions = []
        for key in ["he", "ex"]:
            if d.get(key) is not None:
                mask_arr, _ = self.reader.get_data(self.reader.read(d[key]))
                lesions.append(torch.tensor(mask_arr.astype(np.float32)).squeeze())
            else:
                lesions.append(torch.zeros(spatial, dtype=torch.float32))
        d['label'] = (torch.stack(lesions, dim=0) > 0).float()
        return d

def get_macro_transforms():
    train_tfm = Compose([
        LoadImaged(keys=["image"], reader=PILReader), 
        EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"),
        Resized(keys=["image"], spatial_size=MacroConfig.image_size),
        LoadMacroLesionsd(keys=["he", "ex"]),
        Resized(keys=["label"], spatial_size=MacroConfig.image_size, mode="nearest"),
        RandRotate90d(keys=["image", "label"], prob=0.5, spatial_axes=[0, 1]),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
    ])
    val_tfm = Compose([
        LoadImaged(keys=["image"], reader=PILReader),
        EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"),
        Resized(keys=["image"], spatial_size=MacroConfig.image_size),
        LoadMacroLesionsd(keys=["he", "ex"]),
        Resized(keys=["label"], spatial_size=MacroConfig.image_size, mode="nearest"),
    ])
    return train_tfm, val_tfm

if __name__ == "__main__":
    print(f"Starting Macro-Specialist Training (HE/EX Focus)...")
    train_files = get_monai_data_dicts(MacroConfig.train_images_dir, MacroConfig.train_masks_dir)
    val_files = get_monai_data_dicts(MacroConfig.val_images_dir, MacroConfig.val_masks_dir)
    train_tfm, val_tfm = get_macro_transforms()
    
    train_loader = DataLoader(Dataset(train_files, train_tfm), batch_size=MacroConfig.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(Dataset(val_files, val_tfm), batch_size=1, num_workers=0)

    model = Transfer_UNet(num_classes=MacroConfig.out_channels, backbone=MacroConfig.model_type).to(MacroConfig.device)
    loss_func = HybridLoss(alpha=0.3, beta=0.7)
    optimiser = optim.Adam(model.parameters(), lr=MacroConfig.learning_rate)
    dice_metric = DiceMetric(include_background=True, reduction="mean_batch") 
    scaler = torch.amp.GradScaler('cuda')

    best_dice = -1
    for epoch in range(MacroConfig.num_epochs):
        model.train()
        optimiser.zero_grad()
        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for batch in progress:
            inputs, labels = batch["image"].to(MacroConfig.device).float(), batch["label"].to(MacroConfig.device)
            with torch.amp.autocast('cuda'):
                loss = loss_func(model(inputs), labels)
            scaler.scale(loss).backward()
            scaler.step(optimiser); scaler.update(); optimiser.zero_grad()
            progress.set_postfix({"loss": f"{loss.item():.4f}"})

        model.eval()
        with torch.no_grad():
            for val_batch in val_loader:
                v_out = model(val_batch["image"].to(MacroConfig.device).float())
                dice_metric(y_pred=(torch.sigmoid(v_out) > 0.5).float(), y=val_batch["label"].to(MacroConfig.device))
            
            scores = dice_metric.aggregate() 
            dice_metric.reset()
            mean_dice = scores.mean().item()
            print(f"Epoch {epoch+1} Results: HE: {scores[0]:.4f} | EX: {scores[1]:.4f}")
            if mean_dice > best_dice:
                best_dice = mean_dice
                torch.save(model.state_dict(), "/kaggle/working/best_macro_model.pth")
                print(">>> New Best Model Saved :)")
