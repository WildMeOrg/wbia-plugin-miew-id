import torch
from tqdm import tqdm
import cv2
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

from .pairx.core import explain
from .pairx.xai_dataset import get_pretransform_img
from .helpers import get_chip_from_img, load_image


def draw_one(
    device,
    test_loader,
    model,
    crop_bbox,
    visualization_type="lines_and_colors",
    layer_key="backbone.blocks.3",
    k_lines=20,
    k_colors=10,
):
    """
    Generates a PAIR-X explanation for the provided images and model.

    Args:
        device (str or torch.device): Device to use (cuda or cpu).
        test_loader (DataLoader): Should contain two images, with 4 items for each (image, name, path, bbox as xywh).
        model (torch.nn.Module or equivalent): The deep metric learning model.
        visualization_type (str): The part of the PAIR-X visualization to return, selected from "lines_and_colors" (default), "only_lines", and "only_colors".
        layer_keys (str): The key of the intermediate layer to be used for explanation. Defaults to 'backbone.blocks.3'.
        k_lines (int, optional): The number of matches to visualize as lines. Defaults to 20.
        k_colors (int, optional): The number of matches to backpropagate to original image pixels. Defaults to 10.

    Returns:
        numpy.ndarray: PAIR-X visualization of type visualization_type.
    """
    assert test_loader.batch_size == 1, "test_loader should have a batch size of 1"
    assert len(test_loader) == 2, "test_loader should only contain two images"
    assert visualization_type in (
        "lines_and_colors",
        "only_lines",
        "only_colors",
    ), "unsupported visualization type"

    transformed_images = []
    pretransform_images = []

    # get transformed and untransformed images out of test_loader
    for batch in test_loader:
        (transformed_image,), _, (path,), (bbox,), (theta,) = batch[:5]

        if len(transformed_image.shape) == 3:
            transformed_image = transformed_image.unsqueeze(0)

        transformed_images.append(transformed_image.to(device))

        img_size = tuple(transformed_image.shape[-2:])
        pretransform_image = load_image(path)

        if crop_bbox:
            pretransform_image = get_chip_from_img(pretransform_image, bbox, theta)

        pretransform_image = np.array(transforms.Resize(img_size)(Image.fromarray(pretransform_image)))
        pretransform_images.append(pretransform_image)

    img_0, img_1 = transformed_images
    img_np_0, img_np_1 = pretransform_images

    # If only returning image with lines, skip generating color maps to save time
    if visualization_type == "only_lines":
        k_colors = 0

    # generate explanation image and return
    model.eval()
    model.device = device
    pairx_img = explain(
        img_0,
        img_1,
        img_np_0,
        img_np_1,
        model,
        [layer_key],
        k_lines=k_lines,
        k_colors=k_colors,
    )

    pairx_height = pairx_img.shape[0] // 2

    if visualization_type == "only_lines":
        return pairx_img[:pairx_height]
    elif visualization_type == "only_colors":
        return pairx_img[pairx_height:]

    return pairx_img
