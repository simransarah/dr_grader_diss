import os 
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True" 
import cv2 
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm
from monai.data import DataLoader, Dataset, PILReader
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from monai.transforms import (
    Compose, 
    LoadImaged, 
    EnsureChannelFirstd, 
    ScaleIntensityd, 
    RandRotate90d, 
    RandFlipd, 
    RandCropByPosNegLabeld, 
    MapTransform
)

from segmentation.config import SegmentationConfig
from segmentation.dataset import get_monai_data_dicts
from segmentation.model import Attention_UNet

class EnhanceFundusImaged(MapTransform):
    def __init__(self, keys, gamma=0.8, clip_limit=2.0, tile_grid_size=(8,8), allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)
        self.gamma = gamma
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            img = d[key]
            if isinstance(img, torch.Tensor):
                img = img.numpy()
            
            if img.ndim == 3:
                if img.shape[-1] in [3, 4]:
                    green = img[:, :, 1]
                elif img.shape[0] in [3, 4]:
                    green = img[1, :, :]
                else:
                    green = img.squeeze()
            else:
                green = img 

            if green.max() > 1:
                green = green / 255.0
                
            green = np.power(green, self.gamma)
            
            green_uint8 = (green * 255).astype(np.uint8)
            green_clahe = self.clahe.apply(green_uint8)
            
            green_final = green_clahe.astype(np.float32) / 255.0
            
            d[key] = torch.tensor(green_final).unsqueeze(0)
            
        return d

class LoadLesionsd(MapTransform):
    def __init__(self, keys, allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)
        self.reader = PILReader()

    def __call__(self, data):
        d = dict(data)
        
        spatial_shape = d['image'].shape[1:] 

        lesion_tensors = []
        path_keys = ["ma", "he", "ex"]
        for key in path_keys:
            if d.get(key) is not None:
                mask_obj = self.reader.read(d[key])
                mask_arr, _ = self.reader.get_data(mask_obj)
                
                mask_tsr = torch.tensor(mask_arr.astype(np.float32))
                if mask_tsr.ndim == 3:
                    if mask_tsr.shape[-1] in [3, 4]:
                        mask_tsr = mask_tsr[:, :, 0]
                    elif mask_tsr.shape[0] in [3, 4]:
                        mask_tsr = mask_tsr[0, :, :]
                    else:
                        mask_tsr = mask_tsr.squeeze()
                
                lesion_tensors.append(mask_tsr)
            else:
                lesion_tensors.append(torch.zeros(spatial_shape, dtype=torch.float32))

        label = torch.stack(lesion_tensors, dim=0)
        d['label'] = (label > 0).float()

        for key in path_keys:
            if key in d:
                del d[key] 

        return d

def get_transforms():
    train_tfm = Compose([
        LoadImaged(keys=["image"], reader=PILReader), 
        EnhanceFundusImaged(keys=["image"]),
        LoadLesionsd(keys=["ma", "he", "ex"]),
        
        EnsureChannelFirstd(keys=["image"], channel_dim=0), 

        RandCropByPosNegLabeld(
            keys=["image", "label"],
            label_key="label",
            spatial_size=(512, 512),  
            pos=4, neg=1, 
            num_samples= 16  
        ),
        
        RandRotate90d(keys=["image", "label"], prob=0.5, spatial_axes=[0, 1]),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
    ])

    val_tfm = Compose([
        LoadImaged(keys=["image"], reader=PILReader),
        EnhanceFundusImaged(keys=["image"]),
        LoadLesionsd(keys=["ma", "he", "ex"]),
        EnsureChannelFirstd(keys=["image"], channel_dim=0), 
    ])
    return train_tfm, val_tfm

if __name__ == "__main__":
    print("Starting Green-Channel Training...")
    train_files = get_monai_data_dicts(SegmentationConfig.train_images_dir, SegmentationConfig.train_masks_dir)
    val_files = get_monai_data_dicts(SegmentationConfig.val_images_dir, SegmentationConfig.val_masks_dir)
    
    train_tfm, val_tfm = get_transforms()
    
    train_loader = DataLoader(Dataset(train_files, train_tfm), batch_size=SegmentationConfig.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(Dataset(val_files, val_tfm), batch_size=1, num_workers=0)

    model = Attention_UNet(
        spatial_dims=2,
        in_channels=SegmentationConfig.in_channels,
        out_channels=SegmentationConfig.num_classes,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2)
    ).to(SegmentationConfig.device)
    
    loss_func = DiceCELoss(include_background=True, sigmoid=True, lambda_dice=0.5, lambda_ce=0.5)
    
    optimiser = optim.Adam(model.parameters(), lr=SegmentationConfig.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode='max', factor=0.1, patience=10, verbose=True)
    dice_metric = DiceMetric(include_background=True, reduction="mean_batch") 

    scaler = torch.cuda.amp.GradScaler()
    accumulation_steps = SegmentationConfig.virtual_batch_size // SegmentationConfig.batch_size

    best_dice = -1
    
    for epoch in range(SegmentationConfig.num_epochs):
        model.train()
        optimiser.zero_grad()
        
        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for i, batch in enumerate(progress):
            inputs = batch["image"].to(SegmentationConfig.device).float()
            labels = batch["label"].to(SegmentationConfig.device)
            
            # AMP for memory efficiency
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = loss_func(outputs, labels) / accumulation_steps
            
            scaler.scale(loss).backward()
            
            # Virtual Batch Update
            if (i + 1) % accumulation_steps == 0:
                scaler.step(optimiser)
                scaler.update()
                optimiser.zero_grad()
            
            progress.set_postfix({"loss": f"{loss.item() * accumulation_steps:.4f}"})

        model.eval()
        with torch.no_grad():
            for val_batch in val_loader:
                v_in, v_lab = val_batch["image"].to(SegmentationConfig.device).float(), val_batch["label"].to(SegmentationConfig.device)
                v_out = sliding_window_inference(v_in, SegmentationConfig.patch_size, 4, model)
                v_out = (torch.sigmoid(v_out) > 0.5).float()
                dice_metric(y_pred=v_out, y=v_lab)
            
            scores = dice_metric.aggregate() 
            ma_score = scores[0].item()
            he_score = scores[1].item()
            ex_score = scores[2].item()
            mean_val_dice = scores.mean().item()
            dice_metric.reset()
            scheduler.step(mean_val_dice)
            
            print(f"Epoch {epoch+1} Results:")
            print(f"   MA Dice: {ma_score:.4f}  (Microaneurysms)")
            print(f"   HE Dice: {he_score:.4f}  (Haemorrhages)")
            print(f"   EX Dice: {ex_score:.4f}  (Hard Exudates)")
            print(f"   Mean Dice: {mean_val_dice:.4f}")
            
            if mean_val_dice > best_dice:
                best_dice = mean_val_dice
                torch.save(model.state_dict(), "best_monai_segmentation.pth")
                print(">>> New Best Model Saved :)")