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
    DeleteItemsd
) 

from segmentation.config_micro import MicroConfig
from segmentation.dataset import get_monai_data_dicts
# IMPORT THE NEW MODEL
from segmentation.model import CBAM_AG_UNet

# --- Helper Function for Post-Processing ---
def filter_noise(mask_tensor, min_size=5, max_size=150):
    mask_np = mask_tensor.detach().cpu().numpy().astype(np.uint8)
    mask_np = np.squeeze(mask_np) 
    if mask_np.ndim != 2:
        mask_np = mask_np[0] if mask_np.ndim > 2 else mask_np

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_np, connectivity=8)
    output_mask = np.zeros_like(mask_np)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if min_size < area < max_size:
            output_mask[labels == i] = 1
            
    cleaned_tensor = torch.from_numpy(output_mask).unsqueeze(0).to(mask_tensor.device).float()
    return cleaned_tensor

class EnhanceGreenChanneld(MapTransform):
    def __init__(self, keys, gamma=0.8, clip_limit=2.0, tile_grid_size=(8,8), allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)
        self.gamma, self.clip_limit, self.tile_grid_size = gamma, clip_limit, tile_grid_size

    def __call__(self, data):
        d = dict(data)
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
        
        for key in self.keys:
            img = d[key]
            if isinstance(img, torch.Tensor): img = img.numpy()
            if img.ndim == 3 and img.shape[0] == 3: img = np.transpose(img, (1, 2, 0))
            
            green = img[:, :, 1] if img.ndim == 3 else img 
            if green.max() > 1: green = green / 255.0
            green = np.power(green, self.gamma)
            
            green_uint8 = (green * 255).astype(np.uint8)
            green_clahe = clahe.apply(green_uint8)
            green_final = green_clahe.astype(np.float32) / 255.0
            
            d[key] = torch.tensor(green_final).unsqueeze(0)
        return d

class LoadMAd(MapTransform):
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
        DeleteItemsd(keys=["he", "ex"]), 
        EnhanceGreenChanneld(keys=["image"]),
        LoadMAd(keys=["ma"]),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=MicroConfig.patch_size,  
            pos=2, neg=1, 
            num_samples=4 
        ),
        RandRotate90d(keys=["image", "label"], prob=0.5, spatial_axes=[0, 1]),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
    ])

    val_tfm = Compose([
        LoadImaged(keys=["image"], reader=PILReader),
        DeleteItemsd(keys=["he", "ex"]),
        EnhanceGreenChanneld(keys=["image"]),
        LoadMAd(keys=["ma"]),
    ])
    return train_tfm, val_tfm

if __name__ == "__main__":
    torch.cuda.empty_cache()
    print(f"Starting CBAM-AG-UNet Training (MA Focus)...")
    
    train_files = get_monai_data_dicts(MicroConfig.train_images_dir, MicroConfig.train_masks_dir)
    val_files = get_monai_data_dicts(MicroConfig.val_images_dir, MicroConfig.val_masks_dir)
    train_tfm, val_tfm = get_micro_transforms()
    
    train_loader = DataLoader(Dataset(train_files, train_tfm), batch_size=MicroConfig.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(Dataset(val_files, val_tfm), batch_size=1, num_workers=0)

    # MODEL: New CBAM-AG-UNet
    model = CBAM_AG_UNet(in_channels=1, out_channels=1).to(MicroConfig.device)
    
    # LOSS: Paper uses Cross Entropy (BCEWithLogitsLoss covers this for binary)
    loss_func = torch.nn.BCEWithLogitsLoss()
    
    optimiser = optim.Adam(model.parameters(), lr=MicroConfig.learning_rate)
    
    # SCHEDULER: Paper uses ReduceLROnPlateau
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode='min', factor=0.5, patience=5, min_lr=1e-5
    )

    dice_metric = DiceMetric(include_background=True, reduction="mean_batch") 
    scaler = torch.amp.GradScaler('cuda')

    best_dice = -1
    for epoch in range(MicroConfig.num_epochs):
        model.train()
        optimiser.zero_grad()
        epoch_loss = 0
        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        
        for batch in progress:
            inputs, labels = batch["image"].to(MicroConfig.device).float(), batch["label"].to(MicroConfig.device)
            
            with torch.amp.autocast('cuda'):
                outputs = model(inputs)
                loss = loss_func(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimiser)
            scaler.update()
            optimiser.zero_grad()
            
            epoch_loss += loss.item()
            progress.set_postfix({"loss": f"{loss.item():.4f}"})

        # Update Scheduler based on Loss
        avg_loss = epoch_loss / len(train_loader)
        scheduler.step(avg_loss)

        # Validation
        model.eval()
        with torch.no_grad():
            for val_batch in val_loader:
                v_in = val_batch["image"].to(MicroConfig.device).float()
                v_lab = val_batch["label"].to(MicroConfig.device)
                
                # Sliding window inference
                v_out = sliding_window_inference(v_in, MicroConfig.patch_size, 32, model)
                pred_binary = (torch.sigmoid(v_out) > 0.5).float()
                
                # Post-processing
                pred_cleaned = filter_noise(pred_binary, min_size=5, max_size=150)
                
                # Add batch dim for DiceMetric if needed
                if pred_cleaned.ndim == 3:
                     pred_cleaned = pred_cleaned.unsqueeze(0)

                dice_metric(y_pred=pred_cleaned, y=v_lab)
            
            score = dice_metric.aggregate().item() 
            dice_metric.reset()
            print(f"Epoch {epoch+1} Results: MA Dice: {score:.4f} | Avg Loss: {avg_loss:.4f}")
            
            if score > best_dice:
                best_dice = score
                torch.save(model.state_dict(), "/kaggle/working/best_cbam_model.pth")
                print(">>> New Best Model Saved :)")
