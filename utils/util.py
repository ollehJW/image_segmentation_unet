import h5py
import os
from tqdm import tqdm
import cv2
import numpy as np

def load_data(img_dir, mask_dir = None, exist_mask = True):
    if (img_dir[-3:] == '.pt') and (mask_dir[-3:] == '.pt'):
        images = h5py.File(img_dir, 'r')
        if exist_mask:
            masks = h5py.File(mask_dir, 'r')
    elif (img_dir[-3:] != '.pt') and (mask_dir[-3:] != '.pt'):
        image_files  = sorted(os.listdir(img_dir))
        images = []
        for img in tqdm(image_files):
            images.append(cv2.imread(os.path.join(img_dir, img)))
        images = np.stack(images)
        if exist_mask:
            mask_files  = sorted(os.listdir(mask_dir))
            masks = []
            for mask in tqdm(mask_files):
                masks.append(cv2.imread(os.path.join(mask_dir, mask)))
            masks = np.stack(masks)
    else:
        raise NotImplementedError('Error: Fail to make dataset...')
    
    if exist_mask:
        return dict({"images": images, "masks": masks})
    else:
        return dict({"images": images})
        