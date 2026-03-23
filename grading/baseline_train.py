import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import cohen_kappa_score
 
from grading.config import GradingConfig
 
 
class CombinedDRDataset(Dataset):
    def __init__(self, samples, transform=None):
        # samples is a list of (image_path, label) tuples — normalised from both CSVs upstream
        self.samples = samples
        self.transform = transform
 
    def __len__(self):
        return len(self.samples)
 
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
 
        if self.transform:
            image = self.transform(image)
 
        return image, torch.tensor(label, dtype=torch.long)
 
 
def load_idrid_samples(images_dir, labels_file):
    df = pd.read_csv(labels_file)
    samples = []
    for _, row in df.iterrows():
        path = os.path.join(images_dir, row["Image name"] + ".jpg")
        samples.append((path, int(row["Retinopathy grade"])))
    return samples
 
 
def load_aptos_samples(images_dir, labels_file):
    df = pd.read_csv(labels_file)
    samples = []
    for _, row in df.iterrows():
        path = os.path.join(images_dir, row["id_code"] + ".png")
        samples.append((path, int(row["diagnosis"])))
    return samples
 
 
def build_baseline_model(num_classes):
    # in12k_ft_in1k weights transfer better to fundus images than standard imagenet-1k
    model = timm.create_model("convnext_small.in12k_ft_in1k", pretrained=True, num_classes=num_classes)
    return model
 
 
def compute_class_weights(samples, num_classes):
    counts = np.zeros(num_classes, dtype=np.float32)
    for _, label in samples:
        counts[label] += 1
 
    # fill unrepresented grades with 1 to avoid zero-division
    counts = np.where(counts == 0, 1, counts)
    weights = 1.0 / counts
    weights = weights / weights.sum() * num_classes
    return torch.tensor(weights, dtype=torch.float32)
 
 
def quadratic_weighted_kappa(predicted, targets):
    return cohen_kappa_score(targets, predicted, weights="quadratic")
 
 
def save_checkpoint(state, path):
    torch.save(state, path)
 
 
def load_checkpoint(path, model, optimiser, scaler):
    checkpoint = torch.load(path, map_location=GradingConfig.device)
    model.load_state_dict(checkpoint["model"])
    optimiser.load_state_dict(checkpoint["optimiser"])
    scaler.load_state_dict(checkpoint["scaler"])
    return checkpoint["epoch"], checkpoint["best_qwk"]
 
 
if __name__ == "__main__":
    print(f"Starting Baseline Training using ConvNeXt-Small (in12k_ft_in1k)...")
 
    train_tfm = transforms.Compose([
        transforms.Resize((GradingConfig.img_size, GradingConfig.img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
 
    val_tfm = transforms.Compose([
        transforms.Resize((GradingConfig.img_size, GradingConfig.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
 
    # load and merge both datasets — IDRiD val set stays fixed as the primary benchmark
    idrid_train_samples = load_idrid_samples(GradingConfig.idrid_train_images_dir, GradingConfig.idrid_train_labels_file)
    idrid_val_samples   = load_idrid_samples(GradingConfig.idrid_val_images_dir,   GradingConfig.idrid_val_labels_file)
 
    aptos_samples = load_aptos_samples(GradingConfig.aptos_train_images_dir, GradingConfig.aptos_train_labels_file)
 
    # hold out a portion of APTOS for validation to supplement the small IDRiD val set
    n_aptos_val = int(len(aptos_samples) * GradingConfig.val_split)
    aptos_val_samples   = aptos_samples[:n_aptos_val]
    aptos_train_samples = aptos_samples[n_aptos_val:]
 
    train_samples = idrid_train_samples + aptos_train_samples
    val_samples   = idrid_val_samples   + aptos_val_samples
 
    print(f"Combined train: {len(train_samples)} images  |  val: {len(val_samples)} images")
 
    train_loader = DataLoader(CombinedDRDataset(train_samples, transform=train_tfm), batch_size=GradingConfig.batch_size, shuffle=True,  num_workers=GradingConfig.num_workers, pin_memory=True)
    val_loader   = DataLoader(CombinedDRDataset(val_samples,   transform=val_tfm),   batch_size=GradingConfig.batch_size, shuffle=False, num_workers=GradingConfig.num_workers, pin_memory=True)
 
    model = build_baseline_model(GradingConfig.num_classes).to(GradingConfig.device)
 
    class_weights = compute_class_weights(train_samples, GradingConfig.num_classes).to(GradingConfig.device)
    loss_func = nn.CrossEntropyLoss(weight=class_weights)
 
    optimiser = optim.Adam(model.parameters(), lr=GradingConfig.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimiser, mode='max', factor=0.1, patience=7)
 
    scaler = torch.amp.GradScaler('cuda')
 
    checkpoint_path = os.path.join(GradingConfig.checkpoint_dir, "baseline_checkpoint.pth")
    best_model_path = os.path.join(GradingConfig.checkpoint_dir, "best_baseline_model.pth")
 
    start_epoch = 0
    best_qwk = -1.0
 
    if os.path.exists(checkpoint_path):
        print(f"Resuming from checkpoint: {checkpoint_path}")
        start_epoch, best_qwk = load_checkpoint(checkpoint_path, model, optimiser, scaler)
        print(f"Resumed at epoch {start_epoch + 1}  |  best QWK so far: {best_qwk:.4f}")
 
    for epoch in range(start_epoch, GradingConfig.num_epochs):
        model.train()
        optimiser.zero_grad(set_to_none=True)
 
        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for inputs, labels in progress:
            inputs = inputs.to(GradingConfig.device)
            labels = labels.to(GradingConfig.device)
 
            with torch.amp.autocast('cuda'):
                outputs = model(inputs)
                loss = loss_func(outputs, labels)
 
            scaler.scale(loss).backward()
            scaler.step(optimiser)
            scaler.update()
            optimiser.zero_grad(set_to_none=True)
 
            progress.set_postfix({"loss": f"{loss.item():.4f}"})
 
        model.eval()
        all_preds, all_targets = [], []
 
        with torch.no_grad(), torch.amp.autocast('cuda'):
            for inputs, labels in val_loader:
                inputs = inputs.to(GradingConfig.device)
                outputs = model(inputs)
 
                preds = torch.argmax(torch.softmax(outputs, dim=1), dim=1)
                all_preds.extend(preds.cpu().numpy().tolist())
                all_targets.extend(labels.numpy().tolist())
 
        val_qwk = quadratic_weighted_kappa(all_preds, all_targets)
        scheduler.step(val_qwk)
 
        # save every epoch so a session timeout doesn't lose progress
        save_checkpoint({
            "epoch": epoch + 1,
            "model": model.state_dict(),
            "optimiser": optimiser.state_dict(),
            "scaler": scaler.state_dict(),
            "best_qwk": best_qwk,
        }, checkpoint_path)
 
        print(f"Epoch {epoch+1} Results: QWK: {val_qwk:.4f}")
 
        if val_qwk > best_qwk + 0.001:
            best_qwk = val_qwk
            torch.save(model.state_dict(), best_model_path)
            print(f">>> New Best Baseline Model Saved :)  (QWK: {best_qwk:.4f})")
