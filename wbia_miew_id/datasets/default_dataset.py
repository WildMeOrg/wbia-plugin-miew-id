import os
import cv2
import torch
from torch.utils.data import Dataset
import numpy as np
from .helpers import get_chip_from_img, load_image
import random



class MiewIdDataset(Dataset):
    def __init__(self, csv, transforms=None, fliplr=False, fliplr_view=[], crop_bbox=False, n_train_classes=None):

        self.csv = csv#.reset_index()
        self.augmentations = transforms
        self.fliplr = fliplr
        self.fliplr_view = fliplr_view
        self.crop_bbox = crop_bbox
        self.n_train_classes = n_train_classes

    def __len__(self):
        return self.csv.shape[0]

    def __getitem__(self, index):
        row = self.csv.iloc[index]

        image_path = row['file_path']
        image = load_image(image_path)

        bbox = row['bbox']
        theta = row['theta'] if row['theta'] is not None else 0
        species = row.get('species', 'none')

        if self.crop_bbox:
            image = get_chip_from_img(image, bbox, theta)

        if self.augmentations:
            augmented = self.augmentations(image=image)
            image = augmented['image']

        if self.fliplr:
            if row['viewpoint'] in self.fliplr_view:
                image = torch.from_numpy(np.fliplr(image).copy())

        
        return {"image": image, "label":torch.tensor(row['name']), 
                "image_idx": self.csv.index[index], "file_path": image_path, "bbox": torch.Tensor(bbox),
                'theta': theta, 'species': species}