import albumentations as A
from albumentations.pytorch import ToTensorV2

def image_augmentation(img_size=512):
    # Augmentation pipeline, including CLAHE and Grid Distortion
    return A.Compose([
        A.Resize(height=img_size, width=img_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.CLAHE(p=1.0, clip_limit=4.0, tile_grid_size=(8, 8)),
        A.GridDistortion(p=0.5, num_steps=5, distort_limit=0.3),
        A.Normalize(),
        ToTensorV2()
    ])