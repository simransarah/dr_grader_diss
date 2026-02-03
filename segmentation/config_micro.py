import os 
import torch 

class MicroConfig:
    """
    Configuration for the Micro-Specialist Model (Microaneurysms).
    Focus: High-resolution local features using the Green Channel to maximise contrast.
    """
    # Paths (Kaggle directory structure)
    root_dir = os.path.join("/kaggle", "input", "idriddata", "data", "IDRiD", "A. Segmentation")

    train_images_dir = os.path.join(root_dir, "1. Original Images", "a. Training Set")
    train_masks_dir = os.path.join(root_dir, "2. All Segmentation Groundtruths", "a. Training Set")

    val_images_dir = os.path.join(root_dir, "1. Original Images", "b. Testing Set")
    val_masks_dir = os.path.join(root_dir, "2. All Segmentation Groundtruths", "b. Testing Set")
    
    # Model parameters
    model_type = "attention_unet" # Custom Attention U-Net is best for tiny, sparse targets
    patch_size = (64, 64)         # Critical: Small patches ensure MAs aren't lost in the background
    in_channels = 1               # Green Channel Only (High Contrast)
    out_channels = 1              # Binary Segmentation (MA only)
    
    # Training hyperparameters
    batch_size = 128              # Tiny patches allow for a massive batch size to stabilise gradients
    learning_rate = 5e-4          # Slightly higher LR as we are training from scratch
    num_epochs = 100
    
    # Hardware
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers = 2 

    # Data augmentation parameters
    clahe_probability = 1.0
