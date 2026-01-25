import torch 

def dice_coefficient(predicted, target, smooth=1e-6): 
    # calculates dice score for validation
    # predicted: tensor(B, C, H, W) - can be raw logits or probabilities
    # target: tensor(B, C, H, W) - binary ground truth masks (0 or 1)

    # ensures predictions are [0, 1]
    predicted = torch.sigmoid(predicted)

    # flatten the tensors
    predicted = predicted.view(-1)
    target = target.view(-1)
    intersection = (predicted * target).sum()
    dice = (2. * intersection + smooth) / (predicted.sum() + target.sum() + smooth)
    
    return dice.item()