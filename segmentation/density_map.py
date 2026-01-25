import cv2 
import numpy as np

def create_density_map(binary_masks): 
    # converts binary masks into a 3 channel lesion density map
    # input: binary_masks - tensor of shape (3, H, W) with binary masks for each lesion type [MA, HE, EX]
    # output: density_map - tensor of shape (H, W, 3) so [R=MA, G=HE, B=EX] 

    # the separate channels

    MA = binary_masks[0]
    HE = binary_masks[1]
    EX = binary_masks[2]

    # preserve the spatial information, apply Gaussian blur to each channel which acts as density estimation
    ma_density = cv2.GaussianBlur(MA * 255, (15, 15), 0)
    he_density = cv2.GaussianBlur(HE * 255, (15, 15), 0)
    ex_density = cv2.GaussianBlur(EX * 255, (15, 15), 0)

    # stack the three channels back into a 3-channel density map
    density_map = np.stack([ma_density, he_density, ex_density], axis=2) 

    # normalises to [0, 1] range
    return density_map / 255.0  