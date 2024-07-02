import numpy
import cv2
import glob
import albumentations
from albumentations.core.transforms_interface import ImageOnlyTransform
from albumentations.pytorch.transforms import ToTensorV2

import torchvision.transforms as T
from albumentations.core.transforms_interface import ImageOnlyTransform
from PIL import Image
import numpy as np

class PyTorchResize(ImageOnlyTransform):
    """Wrap PyTorch Resize transform for Albumentations compatibility."""
    def __init__(self, height, width, interpolation=T.InterpolationMode.BILINEAR, always_apply=False, p=1.0):
        super().__init__(always_apply, p)
        self.resize = T.Resize(size=(height, width), interpolation=interpolation)

    def apply(self, img, **params):
        # Convert numpy array to PIL Image
        img_pil = Image.fromarray(img)

        # Apply the resize transform
        img_pil_resized = self.resize(img_pil)

        # Convert back to numpy array
        return np.array(img_pil_resized)


def triangle(img, p):
    xx = numpy.random.rand(1)[0]
    if xx > p:
        h, w, _= img.shape
        limitw = int(w * 0.3)
        limith = int(h * 0.25)
        desc = 0
        step = limitw / limith
        for i in range(limith):
            img[i][:limitw - int(step * i)] = (255, 255, 255)
    return img

class Triangle(ImageOnlyTransform):
    def __init__(self, p):
        super(Triangle, self).__init__(p)
        self.p = p
    def apply(self, img , **params):
        return triangle(img , self.p)

def get_train_transforms(image_size):
    return albumentations.Compose(
        [   
            Triangle(p=0.5),
            PyTorchResize(image_size[0], image_size[1], always_apply=True),
            albumentations.OneOf([
                albumentations.Sharpen(p=0.3),
                albumentations.ToGray(p=0.3),
                albumentations.CLAHE(p=0.3),
            ], p=0.5),
            albumentations.ShiftScaleRotate(
                shift_limit=0.25, scale_limit=0.2, rotate_limit=15, p=0.5
            ),
            albumentations.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            albumentations.Normalize(),
            ToTensorV2(),
        ]
    )

def get_valid_transforms(image_size):
    return albumentations.Compose(
        [
            albumentations.Resize(image_size[0], image_size[1], always_apply=True),
            albumentations.Normalize(),
            ToTensorV2(p=1.0)
        ]
    )

def get_test_transforms(image_size):
    return albumentations.Compose(
        [
            albumentations.Resize(image_size[0], image_size[1], always_apply=True),
            albumentations.Normalize(),
            ToTensorV2(p=1.0)
        ]
    )