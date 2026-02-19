import os 
import cv2
import torch 
import pandas as pd
import numpy as np
from torch.utils.data import Dataset

class GradingDataset(Dataset):
    def __init__(self, maps_dir, labels_csv, transform=None):
        self.maps_dir = maps_dir
        self.transform = transform

        if not os.path.exists(labels_csv):
            raise FileNotFoundError(f"Labels CSV file not found: {labels_csv}")
        self.labels_df = pd.read_csv(labels_csv)

    def __len__(self):
        return len(self.labels_df)
    
    def __getitem__(self, index):
        row = self.labels_df.iloc[index]

        # get filenames
        img_filename = str(row['Image name'])
        if not img_filename.endswith('.jpg'):
            img_filename += '.jpg'

        map_path = os.path.join(self.maps_dir, img_filename)

        # get label
        label = int(row['Retinopathy grade'])

        # load density map
        if os.path.exists(map_path):
            density_map = cv2.imread(map_path, cv2.IMREAD_COLOR) # BGR
            density_map = cv2.cvtColor(density_map, cv2.COLOR_BGR2RGB) # RGB
        else:
            # warn loudly so missing maps don't silently corrupt training
            print(f"[WARNING] Density map not found, using zero placeholder: {map_path}")
            density_map = np.zeros((512, 512, 3), dtype=np.uint8)

        # Albumentations expects 'image' key
        if self.transform is not None:  
            augmented = self.transform(image=density_map)
            density_map = augmented['image']
        else:
            # fallback normalisation
            density_map = torch.from_numpy(density_map).permute(2, 0, 1).float() / 255.0

        return density_map, torch.tensor(label, dtype=torch.long)
