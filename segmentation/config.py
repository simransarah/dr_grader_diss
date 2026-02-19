import os 
import torch 

class SegmentationConfig:
    TARGET_LESION = "ex"
    # paths
    root_dir = os.path.join("/kaggle", "input", "idriddata", "data", "IDRiD", "A. Segmentation")

    train_images_dir = os.path.join(root_dir, "1. Original Images", "a. Training Set")
    train_masks_dir = os.path.join(root_dir, "2. All Segmentation Groundtruths", "a. Training Set")

    val_images_dir = os.path.join(root_dir, "1. Original Images", "b. Testing Set")
    val_masks_dir = os.path.join(root_dir, "2. All Segmentation Groundtruths", "b. Testing Set")
    
    # model parameters
    backbone = "efficientnet-b3"
    pretrained = True
    patch_size = (512, 512)
    in_channels = 3  
    out_channels = 1
    batch_size = 4   
    virtual_batch_size = 16 
    learning_rate = 1e-4
    num_epochs = 100
    num_classes = 3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # data augmentation parameters
    clahe_probability = 1 
    grid_distortion_probability = 0.5
