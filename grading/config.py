import os
import torch
 
class GradingConfig:
    # IDRiD paths
    idrid_root_dir = "/kaggle/input/idriddata/data/IDRiD/B. Disease Grading"
 
    idrid_train_images_dir = os.path.join(idrid_root_dir, "1. Original Images", "a. Training Set")
    idrid_train_labels_file = os.path.join(idrid_root_dir, "2. Groundtruths", "a. IDRiD_Disease Grading_Training Labels.csv")
 
    idrid_val_images_dir = os.path.join(idrid_root_dir, "1. Original Images", "b. Testing Set")
    idrid_val_labels_file = os.path.join(idrid_root_dir, "2. Groundtruths", "b. IDRiD_Disease Grading_Testing Labels.csv")
 
    # APTOS paths — train split is used for training, no official val split so we hold out locally
    aptos_root_dir = "/kaggle/input/aptos2019-blindness-detection"
 
    aptos_train_images_dir = os.path.join(aptos_root_dir, "train_images")
    aptos_train_labels_file = os.path.join(aptos_root_dir, "train.csv")
 
    checkpoint_dir = "/kaggle/working"
 
    # model parameters
    img_size = 512
    batch_size = 4
    learning_rate = 1e-4
    num_epochs = 50
    num_classes = 5
    val_split = 0.1  # fraction of APTOS train set held out for validation
 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers = 2
