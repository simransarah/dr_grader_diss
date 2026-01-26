import torch
import os
import cv2
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from segmentation.model import Attention_UNet
from segmentation.config import SegmentationConfig
from segmentation.dataset import SegmentationDataset
from segmentation.density_map import create_density_map


CHECKPOINT_PATH = "best_model.pth.tar" 
OUTPUT_DIR_TRAIN = os.path.join("data", "IDRiD", "B. Disease Grading", "3. Density Maps", "train")
OUTPUT_DIR_TEST = os.path.join("data", "IDRiD", "B. Disease Grading", "3. Density Maps", "test")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def get_transforms():
    return A.Compose([
        A.Resize(height=SegmentationConfig.img_size, width=SegmentationConfig.img_size),
        A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]),
        ToTensorV2()
    ])

def generate_maps(loader, model, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    model.eval()
    print(f"Generating maps for {output_folder}...")
    
    with torch.no_grad():
        for idx, (images, _) in enumerate(tqdm(loader)):
            images = images.to(DEVICE)
            preds = torch.sigmoid(model(images))
            preds = (preds > 0.5).float()
            
            for i in range(images.shape[0]):
                mask_tensor = preds[i].cpu().numpy()
                d_map = create_density_map(mask_tensor)
                
                # Recover filename from dataset 
                filename = loader.dataset.image_filenames[idx * loader.batch_size + i]
                save_path = os.path.join(output_folder, filename)
                
                # Convert to uint8 and flatten to grayscale (max projection)
                d_map_uint8 = (d_map * 255).astype(np.uint8)
                d_map_gray = np.max(d_map_uint8, axis=2) 
                
                cv2.imwrite(save_path, d_map_gray)

if __name__ == "__main__":
    # Load Model
    model = Attention_UNet(num_classes=SegmentationConfig.num_classes).to(DEVICE)
    checkpoint = torch.load(CHECKPOINT_PATH)
    model.load_state_dict(checkpoint["state_dict"])
    
    # Prepare Data
    transforms = get_transforms()
    train_ds = SegmentationDataset(SegmentationConfig.train_images_dir, SegmentationConfig.train_masks_dir, transform=transforms)
    test_ds = SegmentationDataset(SegmentationConfig.val_images_dir, SegmentationConfig.val_masks_dir, transform=transforms)
    
    # Generate
    generate_maps(DataLoader(train_ds, batch_size=8, shuffle=False), model, OUTPUT_DIR_TRAIN)
    generate_maps(DataLoader(test_ds, batch_size=8, shuffle=False), model, OUTPUT_DIR_TEST)
    print("Done! Maps generated.")