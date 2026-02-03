import os
from torch.utils.data import Dataset

class SegmentationDataset(Dataset):
    def __init__(self, images_dir, masks_dir, transform=None):
        self.data_dicts = get_monai_data_dicts(images_dir, masks_dir)
        self.transform = transform

    def __len__(self):
        return len(self.data_dicts)
    
    def __getitem__(self, index):
        data = self.data_dicts[index]
        if self.transform:
            data = self.transform(data)
        return data

def get_monai_data_dicts(images_dir, masks_dir):
    image_filenames = [img for img in os.listdir(images_dir) if img.endswith('.jpg')]
    data_dicts = []

    for img_name in image_filenames:
        base_name = img_name.replace('.jpg', '')
        
        ma_path = os.path.join(masks_dir, "1. Microaneurysms", base_name + "_MA.tif")
        he_path = os.path.join(masks_dir, "2. Haemorrhages", base_name + "_HE.tif")
        ex_path = os.path.join(masks_dir, "3. Hard Exudates", base_name + "_EX.tif")

        record = {
            "image": os.path.join(images_dir, img_name),
        }
        
        if os.path.exists(ma_path): record["ma"] = ma_path
        if os.path.exists(he_path): record["he"] = he_path
        if os.path.exists(ex_path): record["ex"] = ex_path

        data_dicts.append(record)
        
    return data_dicts