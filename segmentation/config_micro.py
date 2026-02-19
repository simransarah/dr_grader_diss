import os 
import torch 

class MicroConfig:
    """
    Configuration for the Micro-Specialist Model (Microaneurysms).
    Aligned with '12880_2025_Article_1625' Paper Specs.
    """
    # Paths
    root_dir = os.path.join("/kaggle", "input", "idriddata", "data", "IDRiD", "A. Segmentation")
    train_images_dir = os.path.join(root_dir, "1. Original Images", "a. Training Set")
    train_masks_dir = os.path.join(root_dir, "2. All Segmentation Groundtruths", "a. Training Set")
    val_images_dir = os.path.join(root_dir, "1. Original Images", "b. Testing Set")
    val_masks_dir = os.path.join(root_dir, "2. All Segmentation Groundtruths", "b. Testing Set")
    
    # Model parameters
    model_type = "cbam_ag_unet"   # Matches the new model class
    
    # PAPER SPEC: "Patch size 64x64... Dice 0.865"
    patch_size = (64, 64)         
    
    in_channels = 1               # Green Channel Only
    out_channels = 1              # Binary Segmentation
    
    # PAPER SPEC: Batch size 16
    batch_size = 16              
    
    learning_rate = 1e-3          # Initial LR from paper
    num_epochs = 100
    
    # Hardware
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers = 2 

    # Data augmentation parameters
    clahe_probability = 1.0
