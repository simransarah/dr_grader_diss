import torch 
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module): 
    def __init__(self, alpha=0.8, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        # flatten inputs and targets    
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        # calculate standard binary cross entropy loss
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)  # prevents nans when probability is 0

        # calculate focal loss, weighting harder examples more
        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        return focal_loss.mean()
    

class DiceFocalLoss(nn.Module):
    def __init__(self, alpha=0.8, gamma=2, smooth=1):
        super(DiceFocalLoss, self).__init__()
        # uses class FocalLoss defined above
        self.focal_loss = FocalLoss(alpha, gamma)
        self.smooth = smooth

    def forward(self, inputs, targets):
        focal = self.focal_loss(inputs, targets)

        # dice loss calculation
        inputs_prob = torch.sigmoid(inputs)
        inputs_flat = inputs_prob.view(-1)
        targets_flat = targets.view(-1)

        intersection = (inputs_flat * targets_flat).sum()
        dice_loss = 1 - (2. * intersection + self.smooth) / (inputs_flat.sum() + targets_flat.sum() + self.smooth)

        # combine focal loss and dice loss
        return focal + dice_loss