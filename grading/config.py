import os
import torch

class GradingConfig:
    # paths
    root_dir = os.path.join("data", "IDRiD", "B. Disease Grading")

    train_images_dir = os.path.join(root_dir, "1. Original Images", "a. Training Set")
    train_labels_file = os.path.join(root_dir, "2. Groundtruths", "a. IDRiD_Disease Grading_Training Labels.csv")
    train_maps_dir = os.path.join(root_dir, "3. Density Maps", "train")

    val_images_dir = os.path.join(root_dir, "1. Original Images", "b. Testing Set")
    val_labels_file = os.path.join(root_dir, "2. Groundtruths", "b. IDRiD_Disease Grading_Testing Labels.csv")
    val_maps_dir = os.path.join(root_dir, "3. Density Maps", "test")
    # model parameters
    img_size = 512
    batch_size = 4
    learning_rate = 1e-4
    num_epochs = 50
    num_classes = 5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    num_workers = 0