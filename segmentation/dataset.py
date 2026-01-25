import os
import cv2 
import numpy as np
import torch 
from torch.utils.data import Dataset

class SegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.transform = transform

        self.image_filenames = [img for img in os.listdir(images_dir) if img.endswith('.jpg')]

        self.lesion_types = [
            ("1. Microaneurysms", "_MA.tif"), 
            ("2. Haemorrhages", "_HE.tif"), 
            ("3. Hard Exudates", "_EX.tif")
        ]

    def __len__(self):
        return len(self.image_filenames)
    
    def __getitem__(self, index):
        image_filename = self.image_filenames[index]
        image_path = os.path.join(self.images_dir, image_filename)

        # Load image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # create multi-channel mask, starting with all the zeros (background)
        combined_mask = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)

        for i, (folder, suffix) in enumerate(self.lesion_types): 
            mask_filename = image_filename.replace('.jpg', suffix)
            mask_path = os.path.join(self.masks_dir, folder, mask_filename)

            if os.path.exists(mask_path):
                mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    combined_mask[:, :, i] = (mask > 0).astype(np.uint8)

        if self.transform is not None:  
            augmented = self.transform(image=image, mask=combined_mask)
            image = augmented['image']
            combined_mask = augmented['mask']     

        if not isinstance(combined_mask, torch.Tensor):
            combined_mask = torch.from_numpy(combined_mask)

        if combined_mask.ndim == 3 and combined_mask.shape[2] == 3:
            combined_mask = combined_mask.permute(2, 0, 1)  # change from (H, W, C) to (C, H, W)

        return image, combined_mask.float()