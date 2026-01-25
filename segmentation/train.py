import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision.utils import save_image
from segmentation.config import SegmentationConfig
from segmentation.dataset import SegmentationDataset
from segmentation.loss import DiceFocalLoss
from segmentation.metrics import dice_coefficient
from segmentation.model import Attention_UNet

def get_train_transforms():
    return A.Compose([
        A.Resize(height=SegmentationConfig.img_size, width=SegmentationConfig.img_size),
        A.CLAHE(p=SegmentationConfig.clahe_probability),
        A.GridDistortion(p=SegmentationConfig.grid_distortion_probability),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=30,p=0.5), 
        A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]),
        ToTensorV2()
    ])

def get_val_transforms():
    return A.Compose([
        A.Resize(height=SegmentationConfig.img_size, width=SegmentationConfig.img_size),
        A.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]),
        ToTensorV2()
    ])

def save_checkpoint(state, filename="checkpoint.pth.tar"):
    print("Saving checkpoint...")
    torch.save(state, filename)

def save_predictions_as_imgs(loader, model, folder="saved_images/", device="cuda"):
    os.makedirs(folder, exist_ok=True)
    model.eval()
    for index, (x, y) in enumerate(loader):
        x = x.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
        
        # save first prediction in the batch
        save_image(preds, f"{folder}/prediction_{index}.png")
        save_image(y.float(), f"{folder}/ground_truth_{index}.png")

        if index > 2:
            break

    model.train()

def training_loop(loader, model, optimiser, loss_function, scaler): 
    loop = tqdm(loader, leave=True)
    mean_loss = []

    for batch_index, (data, targets) in enumerate(loop):
        data = data.to(device=SegmentationConfig.device)
        targets = targets.to(device=SegmentationConfig.device)

        # forward
        with torch.amp.autocast('cuda'):
            predictions = model(data)
            loss = loss_function(predictions, targets)

        optimiser.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimiser)
        scaler.update()

        mean_loss.append(loss.item())
        # update tqdm loop
        loop.set_postfix(loss=loss.item())

    return sum(mean_loss) / len(mean_loss)

def validate(loader, model, loss_function):
    model.eval()
    dice_score = 0
    num_batches = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=SegmentationConfig.device)
            y = y.to(device=SegmentationConfig.device)

            predictions = model(x)
            dice_score += dice_coefficient(predictions, y)
            num_batches += 1

    model.train()
    score = dice_score / num_batches
    print(f"Validation Dice Score: {score:.4f}")
    return score

if __name__ == "__main__":
    train_transform = get_train_transforms()
    val_transform = get_val_transforms()

    train_dataset = SegmentationDataset(
        images_dir=SegmentationConfig.train_images_dir,
        masks_dir=SegmentationConfig.train_masks_dir,
        transform=train_transform
    )

    val_dataset = SegmentationDataset(
        images_dir=SegmentationConfig.val_images_dir,
        masks_dir=SegmentationConfig.val_masks_dir,
        transform=val_transform
    )

    train_loader = DataLoader(train_dataset, batch_size=SegmentationConfig.batch_size, num_workers=2, pin_memory=True, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=SegmentationConfig.batch_size, num_workers=2, pin_memory=True, shuffle=False)

    model = Attention_UNet(num_classes=SegmentationConfig.num_classes).to(SegmentationConfig.device)
    loss_function = DiceFocalLoss(alpha=0.8, gamma=2)
    optimiser = optim.Adam(model.parameters(), lr=SegmentationConfig.learning_rate)
    scaler = torch.amp.GradScaler('cuda')

    best_dice_score = 0

    for epoch in range(SegmentationConfig.num_epochs):
        print(f"Epoch [{epoch+1}/{SegmentationConfig.num_epochs}]")
        training_loop(train_loader, model, optimiser, loss_function, scaler)
        current_score = validate(val_loader, model, loss_function)

        if current_score > best_dice_score:
            best_dice_score = current_score
            save_checkpoint({
                'state_dict': model.state_dict(),
                'optimiser': optimiser.state_dict(),
            }, filename="best_model.pth.tar")
            save_predictions_as_imgs(val_loader, model)

        print(f"Best Dice Score so far: {best_dice_score:.4f}")