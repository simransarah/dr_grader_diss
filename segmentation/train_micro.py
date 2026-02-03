import os 
import cv2 
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
from monai.data import DataLoader, Dataset, PILReader
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from monai.transforms import (
    Compose, 
    LoadImaged, 
    RandRotate90d, 
    RandFlipd, 
    RandCropByPosNegLabeld, 
    MapTransform,
    DeleteKeysd
) 

from segmentation.config_micro import MicroConfig
from segmentation.dataset import get_monai_data_dicts
from segmentation.model import Attention_UNet
from segmentation.loss import HybridLoss

class EnhanceGreenChanneld(MapTransform):
    """
    Extracts the Green Channel and applies CLAHE + Gamma Correction.
    Returns a single-channel tensor (1, H, W) to minimise spectral noise.
    """
    def __init__(self, keys, gamma=0.8, clip_limit=2.0, tile_grid_size=(8,8), allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)
        self.gamma, self.clip_limit, self.tile_grid_size = gamma, clip_limit, tile_grid_size

    def __call__(self, data):
        d = dict(data)
        # Initialise CLAHE here for multi-processing compatibility
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
        
        for key in self.keys:
            img = d[key]
            if isinstance(img, torch.Tensor): img = img.numpy()
            if img.ndim == 3 and img.shape[0] == 3: img = np.transpose(img, (1, 2, 0))
            
            # 1. Extract Green Channel (Index 1 in RGB)
            green = img[:, :, 1] if img.ndim == 3 else img 

            # 2. Normalisation
            if green.max() > 1: green = green / 255.0
                
            # 3. Gamma Correction (Darkens background to highlight bright MAs)
            green = np.power(green, self.gamma)
            
            # 4. CLAHE (Local Contrast Enhancement)
            green_uint8 = (green * 255).astype(np.uint8)
            green_clahe = clahe.apply(green_uint8)
            green_final = green_clahe.astype(np.float32) / 255.0
            
            # Return as Single Channel Tensor (1, H, W)
            d[key] = torch.tensor(green_final).unsqueeze(0)
            
        return d

class LoadMAd(MapTransform):
    """
    Specialised loader that only retrieves Microaneurysm (MA) masks.
    """
    def __init__(self, keys, allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)
        self.reader = PILReader()

    def __call__(self, data):
        d = dict(data)
        if d.get("ma") is not None:
            mask_arr, _ = self.reader.get_data(self.reader.read(d["ma"]))
            mask_tsr = torch.tensor(mask_arr.astype(np.float32))
            if mask_tsr.ndim == 3: mask_tsr = mask_tsr.squeeze()
            d['label'] = (mask_tsr > 0).float().unsqueeze(0)
        else:
            spatial = d['image'].shape[1:] 
            d['label'] = torch.zeros((1, *spatial), dtype=torch.float32)
        return d

def get_micro_transforms():
    train_tfm = Compose([
        LoadImaged(keys=["image"], reader=PILReader), 
        DeleteKeysd(keys=["he", "ex"], allow_missing_keys=True), 
        
        EnhanceGreenChanneld(keys=["image"]),
        LoadMAd(keys=["ma"]), 
        
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=MicroConfig.patch_size,  
            pos=2, neg=1, 
            num_samples=16 
        ),
        
        RandRotate90d(keys=["image", "label"], prob=0.5, spatial_axes=[0, 1]),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
    ])

    val_tfm = Compose([
        LoadImaged(keys=["image"], reader=PILReader),
        DeleteKeysd(keys=["he", "ex"], allow_missing_keys=True),
        EnhanceGreenChanneld(keys=["image"]),
        LoadMAd(keys=["ma"]),
    ])
    return train_tfm, val_tfm

if __name__ == "__main__":
    print(f"Starting Micro-Specialist Training (MA Focus)...")
    train_files = get_monai_data_dicts(MicroConfig.train_images_dir, MicroConfig.train_masks_dir)
    val_files = get_monai_data_dicts(MicroConfig.val_images_dir, MicroConfig.val_masks_dir)
    train_tfm, val_tfm = get_micro_transforms()
    
    train_loader = DataLoader(Dataset(train_files, train_tfm), batch_size=MicroConfig.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(Dataset(val_files, val_tfm), batch_size=1, num_workers=0)

    model = Attention_UNet(in_channels=1, out_channels=1).to(MicroConfig.device)
    loss_func = HybridLoss(alpha=0.3, beta=0.7)
    optimiser = optim.Adam(model.parameters(), lr=MicroConfig.learning_rate)
    dice_metric = DiceMetric(include_background=True, reduction="mean_batch") 
    scaler = torch.amp.GradScaler('cuda')

    best_dice = -1
    for epoch in range(MicroConfig.num_epochs):
        model.train()
        optimiser.zero_grad()
        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for batch in progress:
            inputs, labels = batch["image"].to(MicroConfig.device).float(), batch["label"].to(MicroConfig.device)
            with torch.amp.autocast('cuda'):
                loss = loss_func(model(inputs), labels)
            scaler.scale(loss).backward()
            scaler.step(optimiser); scaler.update(); optimiser.zero_grad()
            progress.set_postfix({"loss": f"{loss.item():.4f}"})

        model.eval()
        with torch.no_grad():
            for val_batch in val_loader:
                v_out = sliding_window_inference(val_batch["image"].to(MicroConfig.device).float(), MicroConfig.patch_size, 32, model)
                dice_metric(y_pred=(torch.sigmoid(v_out) > 0.5).float(), y=val_batch["label"].to(MicroConfig.device))
            
            score = dice_metric.aggregate().item() 
            dice_metric.reset()
            print(f"Epoch {epoch+1} Results: MA Dice: {score:.4f}")
            if score > best_dice:
                best_dice = score
                torch.save(model.state_dict(), "/kaggle/working/best_micro_model.pth")
                print(">>> New Best Model Saved :)")
