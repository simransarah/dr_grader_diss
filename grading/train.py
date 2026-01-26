import torch
import torch.nn as nn 
import torch.optim as optim 
from torch.utils.data import DataLoader
from tqdm import tqdm 
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os 
from grading.config import GradingConfig
from grading.dataset import GradingDataset
from grading.model import HybridModel
from grading.metrics import quadratic_weighted_kappa

def get_transforms():
    return A.Compose([
        A.Resize(height=GradingConfig.img_size, width=GradingConfig.img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

def train_one_epoch(loader, model, optimiser, loss_function, scaler):
    loop = tqdm(loader, leave=True)
    total_loss = 0

    for batch_index, (data, targets) in enumerate(loop):
        data = data.to(GradingConfig.device)
        targets = targets.to(device=GradingConfig.device)

        # forward
        with torch.amp.autocast('cuda'):
            predictions = model(data)
            loss = loss_function(predictions, targets)

        # backward
        optimiser.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimiser)
        scaler.update()

        total_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(loader)
    return avg_loss

def validate(loader, model, loss_function):
    model.eval()
    all_predictions = []
    all_targets = []
    val_loss = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=GradingConfig.device)
            y = y.to(device=GradingConfig.device)

            predictions = model(x)
            loss = loss_function(predictions, y)
            val_loss += loss.item()
            all_predictions.append(predictions)
            all_targets.append(y)

    qwk = quadratic_weighted_kappa(torch.cat(all_predictions), torch.cat(all_targets))
    model.train()
    print(f"Validation Loss: {val_loss / len(loader):.4f}, QWK: {qwk:.4f}")
    return qwk 

if __name__ == "__main__":
    transforms = get_transforms()

    train_dataset = GradingDataset(GradingConfig.train_images_dir, 
                                   GradingConfig.train_maps_dir,
                                   GradingConfig.train_labels_file,
                                   transforms)
    validation_dataset = GradingDataset(GradingConfig.val_images_dir,
                                        GradingConfig.val_maps_dir,
                                        GradingConfig.val_labels_file,
                                        transforms)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=GradingConfig.batch_size, 
        shuffle=True, 
        num_workers=GradingConfig.num_workers,
        pin_memory=True
    )
    validation_loader = DataLoader(
        validation_dataset, 
        batch_size=GradingConfig.batch_size, 
        shuffle=False, 
        num_workers=GradingConfig.num_workers,
        pin_memory=True
    )
    
    model = HybridModel(num_classes=GradingConfig.num_classes).to(GradingConfig.device)
    loss_function = nn.CrossEntropyLoss()
    optimiser = optim.Adam(model.parameters(), lr=GradingConfig.learning_rate)
    scaler = torch.amp.GradScaler('cuda')

    print("Starting Stage 2 training process...")
    best_qwk = -1
    for epoch in range(GradingConfig.num_epochs):
        print(f"Epoch [{epoch+1}/{GradingConfig.num_epochs}]")
        train_one_epoch(train_loader, model, optimiser, loss_function, scaler)
        qwk = validate(validation_loader, model, loss_function)
        if qwk > best_qwk:
            best_qwk = qwk
            torch.save(model.state_dict(), "best_grading_model.pth")
            print("Saved Best Model!")