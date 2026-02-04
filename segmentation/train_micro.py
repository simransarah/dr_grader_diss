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
from segmentation.model import Attention_UNet
from segmentation.loss import HybridLoss

# --- Helper Function for Post-Processing ---
def filter_noise(mask_tensor, min_size=5, max_size=150):
    """
    Removes connected components smaller than min_size or larger than max_size.
    Expects mask_tensor to be (B, C, H, W) or (C, H, W).
    """
    # 1. Move to CPU and convert to numpy
    mask_np = mask_tensor.detach().cpu().numpy().astype(np.uint8)
    
    # 2. Force it into a 2D shape (H, W) by removing all singleton dimensions
    # This is critical for OpenCV compatibility
    mask_np = np.squeeze(mask_np) 
    
    if mask_np.ndim != 2:
        # If it's still not 2D (e.g. multi-channel), take the first channel
        mask_np = mask_np[0] if mask_np.ndim > 2 else mask_np

    # 3. Run connected components
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask_np, connectivity=8)
    
    output_mask = np.zeros_like(mask_np)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if min_size < area < max_size:
            output_mask[labels == i] = 1
            
    # 4. Convert back to Tensor and restore (C, H, W) format for MONAI metrics
    # MONAI DiceMetric usually expects [Batch, Channel, H, W] or [Channel, H, W]
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
    # --- RESTART KERNEL FIX ---
    torch.cuda.empty_cache()
    # --------------------------

    print(f"Starting Micro-Specialist Training (MA Focus)...")
    train_files = get_monai_data_dicts(MicroConfig.train_images_dir, MicroConfig.train_masks_dir)
    val_files = get_monai_data_dicts(MicroConfig.val_images_dir, MicroConfig.val_masks_dir)
    train_tfm, val_tfm = get_micro_transforms()
    
    train_loader = DataLoader(Dataset(train_files, train_tfm), batch_size=MicroConfig.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(Dataset(val_files, val_tfm), batch_size=1, num_workers=0)

    model = Attention_UNet(in_channels=1, out_channels=1).to(MicroConfig.device)
    loss_func = HybridLoss(alpha=0.3, beta=0.7)
    optimiser = optim.Adam(model.parameters(), lr=MicroConfig.learning_rate)
    
    # Gradient Accumulation Calculation
    accumulation_steps = MicroConfig.virtual_batch_size // MicroConfig.batch_size
    
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimiser, 
        max_lr=1e-3, 
        epochs=MicroConfig.num_epochs, 
        steps_per_epoch=len(train_loader) // accumulation_steps # Adjust steps for accumulation
    )

    dice_metric = DiceMetric(include_background=True, reduction="mean_batch") 
    scaler = torch.amp.GradScaler('cuda')

    best_dice = -1
    for epoch in range(MicroConfig.num_epochs):
        model.train()
        optimiser.zero_grad()
        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        
        # Training Loop
        for i, batch in enumerate(progress):
            inputs, labels = batch["image"].to(MicroConfig.device).float(), batch["label"].to(MicroConfig.device)
            
            with torch.amp.autocast('cuda'):
                outputs = model(inputs)
                # Normalize loss by accumulation steps
                loss = loss_func(outputs, labels) / accumulation_steps
            
            scaler.scale(loss).backward()
            
            # --- GRADIENT ACCUMULATION STEP ---
            if (i + 1) % accumulation_steps == 0:
                scaler.step(optimiser)
                scaler.update()
                scheduler.step()
                optimiser.zero_grad()
            # ----------------------------------
            
            # Multiply loss back for display purposes
            progress.set_postfix({"loss": f"{loss.item() * accumulation_steps:.4f}"})

        # Validation Loop
        model.eval()
        with torch.no_grad():
            for val_batch in val_loader:
                v_in = val_batch["image"].to(MicroConfig.device).float()
                v_lab = val_batch["label"].to(MicroConfig.device)
                
                # Inference
                v_out = sliding_window_inference(v_in, MicroConfig.patch_size, 32, model)
                pred_binary = (torch.sigmoid(v_out) > 0.5).float()
                
                # Apply noise filter (Process one image at a time if batch > 1)
                # Since your val_loader batch_size is 1, pred_binary[0] is safe
                pred_cleaned = filter_noise(pred_binary[0], min_size=5, max_size=150)
                
                # Add batch dimension back for the DiceMetric
                dice_metric(y_pred=pred_cleaned.unsqueeze(0), y=v_lab)
            
            score = dice_metric.aggregate().item() 
            dice_metric.reset()
            print(f"Epoch {epoch+1} Results: MA Dice: {score:.4f}")
            
            if score > best_dice:
                best_dice = score
                torch.save(model.state_dict(), "/kaggle/working/best_micro_model.pth")
                print(">>> New Best Model Saved :)")
