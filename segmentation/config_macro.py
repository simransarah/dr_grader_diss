import os 
import torch 

class MacroConfig:
    """
    Configuration for the Macro-Specialist Model (Hemorrhages & Exudates).
    Focus: Global context (resized full image) to capture large clusters.
    """
    root_dir = os.path.join("/kaggle", "input", "idriddata", "data", "IDRiD", "A. Segmentation")
    train_images_dir = os.path.join(root_dir, "1. Original Images", "a. Training Set")
    train_masks_dir = os.path.join(root_dir, "2. All Segmentation Groundtruths", "a. Training Set")
    val_images_dir = os.path.join(root_dir, "1. Original Images", "b. Testing Set")
    val_masks_dir = os.path.join(root_dir, "2. All Segmentation Groundtruths", "b. Testing Set")
    
    model_type = "efficientnet-b3"
    image_size = (1024, 1024)     # Resizing to 1k preserves global context for EX clusters
    in_channels = 3               # RGB to distinguish Yellow (EX) from Red (HE)
    out_channels = 2              # HE and EX only
    
    batch_size = 4
    learning_rate = 1e-4
    num_epochs = 100
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
