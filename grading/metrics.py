import torch
from sklearn.metrics import cohen_kappa_score

def quadratic_weighted_kappa(predicted, targets): 

    probabilities = torch.softmax(predicted, dim=1)
    predictions = torch.argmax(probabilities, dim=1).cpu().numpy()
    targets = targets.cpu().numpy()

    kappa = cohen_kappa_score(targets, predictions, weights='quadratic')
    return kappa