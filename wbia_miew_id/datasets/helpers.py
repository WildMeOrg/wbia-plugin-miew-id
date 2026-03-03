## Functions for handling rotated bounding boxes

import numpy as np
from PIL import Image as PILImage
import matplotlib.pyplot as plt


def load_image(image_path):
    image = np.array(PILImage.open(image_path).convert('RGB'))
    return image

def imshow(img):
    plt.figure(figsize=(12, 8))
    plt_img = img.copy()
    plt.imshow(plt_img)
    plt.show()

def show_image(image_path):
    image = load_image(image_path)
    imshow(image)

def rotate_box(x1,y1,x2,y2,theta):
    xm = (x1 + x2) // 2
    ym = (y1 + y2) // 2

    h = int(y2 - y1)
    w = int(x2 - x1)

    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])

    A = np.array([[x1, y1], [x1, y2], [x2, y2], [x2, y1], [x1, y1]])
    C = np.array([[xm, ym]])
    RA = (A - C) @ R.T + C
    RA = RA.astype(int)

    return RA

def crop_rect(img, rect):
    """Crop a rotated rectangle from an image using PIL (no OpenCV)."""
    center, size, angle = rect[0], rect[1], rect[2]
    center, size = tuple(map(int, center)), tuple(map(int, size))

    height, width = img.shape[0], img.shape[1]

    diag_len = int(np.sqrt(height * height + width * width))
    new_width = diag_len
    new_height = diag_len

    # Create white canvas and paste image centered
    blank_canvas = np.ones((new_height, new_width, 3), dtype=img.dtype) * 255

    x_offset = (new_width - width) // 2
    y_offset = (new_height - height) // 2

    blank_canvas[y_offset:y_offset+height, x_offset:x_offset+width] = img

    # Convert to PIL for rotation
    pil_canvas = PILImage.fromarray(blank_canvas)

    # PIL rotate uses degrees, counter-clockwise positive (same as cv2.getRotationMatrix2D)
    angle_deg = np.rad2deg(angle)
    new_center_x = new_width // 2
    new_center_y = new_height // 2

    # Rotate around center of canvas
    pil_rotated = pil_canvas.rotate(angle_deg, resample=PILImage.BILINEAR, center=(new_center_x, new_center_y), fillcolor=(255, 255, 255))

    img_rot = np.array(pil_rotated)

    # Compute where the crop center moved to after rotation
    cos_a = np.cos(np.deg2rad(angle_deg))
    sin_a = np.sin(np.deg2rad(angle_deg))
    # Point relative to rotation center
    px = center[0] + x_offset - new_center_x
    py = center[1] + y_offset - new_center_y
    # Apply rotation (counter-clockwise)
    new_cx = cos_a * px + sin_a * py + new_center_x
    new_cy = -sin_a * px + cos_a * py + new_center_y

    # Crop sub-rectangle centered at (new_cx, new_cy) with given size
    crop_w, crop_h = size
    x1 = int(new_cx - crop_w / 2)
    y1 = int(new_cy - crop_h / 2)
    x2 = x1 + crop_w
    y2 = y1 + crop_h

    # Clamp to image bounds
    x1c = max(0, x1)
    y1c = max(0, y1)
    x2c = min(new_width, x2)
    y2c = min(new_height, y2)

    img_crop = img_rot[y1c:y2c, x1c:x2c]

    # Pad if crop extends beyond canvas
    if img_crop.shape[0] != crop_h or img_crop.shape[1] != crop_w:
        padded = np.ones((crop_h, crop_w, 3), dtype=img.dtype) * 255
        paste_x = x1c - x1
        paste_y = y1c - y1
        padded[paste_y:paste_y+img_crop.shape[0], paste_x:paste_x+img_crop.shape[1]] = img_crop
        img_crop = padded

    return img_crop, img_rot


def get_chip_from_img(img, bbox, theta):
    x1,y1,w,h = bbox
    x2 = x1 + w
    y2 = y1 + h
    xm = (x1 + x2) // 2
    ym = (y1 + y2) // 2

    # Treat NaN/None/invalid theta as zero rotation
    try:
        theta_valid = float(theta)
        if theta_valid != theta_valid:  # NaN check
            theta_valid = 0
    except (TypeError, ValueError):
        theta_valid = 0
    theta = theta_valid

    # Do a faster, regular crop if theta is negligible
    if abs(theta) < 0.1:
        x1, y1, w, h = [max(0, int(x)) for x in bbox]
        cropped_image = img[y1 : y1 + h, x1 : x1 + w]
    else:
        cropped_image = crop_rect(img, ((xm, ym), (x2-x1, y2-y1), theta))[0]

    if min(cropped_image.shape) < 1:
        # Use original image
        print(f'Using original image. Invalid parameters - theta: {theta}, bbox: {bbox}')
        cropped_image = img

    return cropped_image
