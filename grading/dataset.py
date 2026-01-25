import os 
import cv2
import torch 
import pandas as pd
from torch.utils.data import Dataset

class GradingDataset(Dataset):
    def __init__(self, images_dir, maps_dir, labels_csv, transform=None):
        self.images_dir = images_dir
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
        img_filename = row['Image Name']
        if not img_filename.endswith('.jpg'):
            img_filename += '.jpg'

        img_path = os.path.join(self.images_dir, img_filename)
        map_path = os.path.join(self.maps_dir, img_filename)

        # get label
        label = int(row['Disease Grade'])

        # load image
        image = cv2.imread(img_path)
        if image is None:
            image = np.zeros((512, 512, 3), dtype=np.uint8)  # placeholder for missing image
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # load density map
        if os.path.exists(map_path):
            density_map = cv2.imread(map_path, cv2.IMREAD_GRAYSCALE)
        else:
            density_map = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)  # placeholder for missing map

        # resize density map if dimensions don't match
        if density_map.shape != (image.shape[0], image.shape[1]):
            density_map = cv2.resize(density_map, (image.shape[1], image.shape[0]))

        
        # augmentation 
        if self.transform is not None:  
            augmented = self.transform(image=image, mask=density_map)
            image = augmented['image']
            density_map = augmented['mask']

            # normalise density map to [0, 1] and add channel dimension
            density_map = density_map.float().unsqueeze(0) / 255.0

        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0 
            density_map = torch.from_numpy(density_map).unsqueeze(0).float() / 255.0

        fused_input = torch.cat([image, density_map], dim=0)  
        
        return fused_input, torch.tensor(label, dtype=torch.long)