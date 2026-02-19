import os 
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True" 
import cv2 
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
import segmentation_models_pytorch as smp

from monai.data import DataLoader, Dataset, PILReader
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from monai.losses import DiceFocalLoss
from monai.transforms import (
    Compose, 
    LoadImaged, 
    RandRotate90d, 
    RandFlipd, 
    RandCropByPosNegLabeld, 
    MapTransform
)

from segmentation.config import SegmentationConfig
from segmentation.dataset import get_monai_data_dicts

class EnhanceFundusImaged(MapTransform):
    def __init__(self, keys, gamma=0.8, clip_limit=2.0, tile_grid_size=(8,8), allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)
        self.gamma = gamma
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.lut = np.array([((i / 255.0) ** gamma) * 255 for i in range(256)]).astype("uint8")

    def __call__(self, data):
        d = dict(data)
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid_size)
        
        for key in self.keys:
            img = d[key]
            if isinstance(img, torch.Tensor):
                img = img.numpy()
            
            if img.ndim == 3 and img.shape[0] == 3:
                img = np.transpose(img, (1, 2, 0))
            
            if img.ndim == 3:
                green = img[:, :, 1]
            else:
                green = img 

            if green.max() > 1.0:
                green_uint8 = (green * 255).astype(np.uint8)
            else:
                green_uint8 = green.astype(np.uint8)

            green_gamma = cv2.LUT(green_uint8, self.lut)
            green_clahe = clahe.apply(green_gamma)
            green_final = green_clahe.astype(np.float32) / 255.0
            
            if img.ndim == 3 and img.shape[-1] == 3:
                img[:, :, 1] = green_final
            else:
                img = np.stack([green_final, green_final, green_final], axis=-1)

            d[key] = torch.tensor(np.transpose(img, (2, 0, 1)))
            
        return d

class LoadTargetLesiond(MapTransform):
    def __init__(self, keys, target_lesion="ex", allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)
        self.reader = PILReader()
        self.target_lesion = target_lesion

    def __call__(self, data):
        d = dict(data)
        spatial_shape = d['image'].shape[1:] 

        # Only load the targeted lesion
        if d.get(self.target_lesion) is not None:
            mask_obj = self.reader.read(d[self.target_lesion])
            mask_arr, _ = self.reader.get_data(mask_obj)
            mask_tsr = torch.tensor(mask_arr.astype(np.float32))
            
            if mask_tsr.ndim == 3:
                if mask_tsr.shape[-1] in [3, 4]:
                    mask_tsr = mask_tsr[:, :, 0]
                elif mask_tsr.shape[0] in [3, 4]:
                    mask_tsr = mask_tsr[0, :, :]
                else:
                    mask_tsr = mask_tsr.squeeze()
        else:
            mask_tsr = torch.zeros(spatial_shape, dtype=torch.float32)

        # Add channel dimension to make it [1, H, W]
        d['label'] = (mask_tsr.unsqueeze(0) > 0).float()

        # Delete path keys so MONAI DataLoader doesn't crash trying to load them
        for key in ["ma", "he", "ex"]:
            if key in d:
                del d[key] 

        return d

def get_transforms(target_lesion):
    # Dynamically adjust sampling rates based on lesion sparsity
    if target_lesion == "ma":
        pos_ratio = 9
    elif target_lesion == "he":
        pos_ratio = 3
    else: # "ex"
        pos_ratio = 1

    train_tfm = Compose([
        LoadImaged(keys=["image"], reader=PILReader), 
        EnhanceFundusImaged(keys=["image"]),
        LoadTargetLesiond(keys=["ma", "he", "ex"], target_lesion=target_lesion),
        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=SegmentationConfig.patch_size,  
            pos=pos_ratio, neg=1, 
            num_samples=4 
        ),
        RandRotate90d(keys=["image", "label"], prob=0.5, spatial_axes=[0, 1]),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
    ])

    val_tfm = Compose([
        LoadImaged(keys=["image"], reader=PILReader),
        EnhanceFundusImaged(keys=["image"]),
        LoadTargetLesiond(keys=["ma", "he", "ex"], target_lesion=target_lesion),
    ])
    return train_tfm, val_tfm


if __name__ == "__main__":
    target = SegmentationConfig.TARGET_LESION
    print(f"Starting Specialist Training for: {target.upper()} using {SegmentationConfig.backbone}...")
    
    train_files = get_monai_data_dicts(SegmentationConfig.train_images_dir, SegmentationConfig.train_masks_dir)
    val_files = get_monai_data_dicts(SegmentationConfig.val_images_dir, SegmentationConfig.val_masks_dir)
    
    train_tfm, val_tfm = get_transforms(target)
    
    train_loader = DataLoader(Dataset(train_files, train_tfm), batch_size=SegmentationConfig.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(Dataset(val_files, val_tfm), batch_size=1, num_workers=0)

    # 1. Initialize Pre-trained SMP Model
    model = smp.Unet(
        encoder_name=SegmentationConfig.backbone, 
        encoder_weights="imagenet" if SegmentationConfig.pretrained else None,     
        in_channels=3,
        classes=1,                      
        decoder_attention_type="scse"   
    ).to(SegmentationConfig.device)
    
    # 2. Initialize Stable MONAI Loss with Dynamic Gamma
    gamma_val = 3.0 if target in ["ma", "he"] else 2.0
    loss_func = DiceFocalLoss(
        include_background=False, 
        sigmoid=True,             
        squared_pred=True,        
        gamma=gamma_val
    )
    
    optimiser = optim.Adam(model.parameters(), lr=SegmentationConfig.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode='max', factor=0.1, patience=10)
    dice_metric = DiceMetric(include_background=False, reduction="mean_batch") 

    scaler = torch.amp.GradScaler('cuda')
    accumulation_steps = SegmentationConfig.virtual_batch_size // SegmentationConfig.batch_size

    best_dice = -1
    
    for epoch in range(SegmentationConfig.num_epochs):
        model.train()
        optimiser.zero_grad()
        
        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for i, batch in enumerate(progress):
            inputs = batch["image"].to(SegmentationConfig.device).float()
            labels = batch["label"].to(SegmentationConfig.device)
            
            with torch.amp.autocast('cuda'):
                outputs = model(inputs)
                loss = loss_func(outputs, labels) / accumulation_steps
            
            scaler.scale(loss).backward()
            
            if (i + 1) % accumulation_steps == 0:
                scaler.step(optimiser)
                scaler.update()
                optimiser.zero_grad()
            
            progress.set_postfix({"loss": f"{loss.item() * accumulation_steps:.4f}"})

        model.eval()
        with torch.no_grad():
            for val_batch in val_loader:
                v_in, v_lab = val_batch["image"].to(SegmentationConfig.device).float(), val_batch["label"].to(SegmentationConfig.device)
                v_out = sliding_window_inference(v_in, SegmentationConfig.patch_size, 4, model, overlap = 0.5)
                v_out = (torch.sigmoid(v_out) > 0.5).float()
                dice_metric(y_pred=v_out, y=v_lab)
            
            scores = dice_metric.aggregate() 
            dice_metric.reset()
            
            val_dice = scores.item()
            scheduler.step(val_dice)
            
            print(f"Epoch {epoch+1} Results: {target.upper()} Dice: {val_dice:.4f}")
            
            if val_dice > best_dice:
                best_dice = val_dice
                # Save the model with the target lesion name
                torch.save(model.state_dict(), f"best_{target}_model.pth")
                print(f">>> New Best {target.upper()} Model Saved :)")
