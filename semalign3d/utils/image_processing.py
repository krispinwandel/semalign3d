from typing import Union
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2


def erode_mask(mask: np.ndarray, kernel_size=2, iterations=2):
    # Define the kernel
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # Apply erosion
    eroded_mask = np.array(
        cv2.erode(mask.astype(np.uint8) * 255, kernel, iterations=iterations)
    ).astype(bool)
    return eroded_mask


def get_pad_sizes(x):
    max_size = max(x.shape[0], x.shape[1])
    return max_size - x.shape[0], max_size - x.shape[1]


def get_pad_sizes_from_img_shape(h, w):
    max_size = max(h, w)
    return max_size - h, max_size - w


def create_coordinate_tensor(h, w, device=None):
    """
    Args:
        h: height of the image
        w: width of the image
        device: device to create the tensor on (default: None, which uses the current device)
    Returns:
        coordinates: (h, w, 2) tensor where coordinates[i, j] = (j, i)
        i.e. coordinates are in (x, y) format
    """
    # Create coordinate tensors
    y_coords, x_coords = torch.meshgrid(
        torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij"
    )
    # Stack them together to create a tensor of shape (h, w, 2)
    coordinates = torch.stack((x_coords, y_coords), dim=-1)
    return coordinates


def img_coords_to_embd_coords(img_coords_xy: torch.Tensor, img_shape, embd_size: int):
    h_pad, w_pad = get_pad_sizes_from_img_shape(*img_shape)
    img_size = max(img_shape)
    pad = torch.tensor(
        [w_pad // 2, h_pad // 2], device=img_coords_xy.device, dtype=img_coords_xy.dtype
    )
    embd_coords_xy = ((img_coords_xy + pad).float() / img_size * embd_size).long()
    embd_coords_xy = torch.clamp(embd_coords_xy, 0, embd_size - 1)
    return embd_coords_xy


def embd_coords_to_img_coords(embd_coords_xy: torch.Tensor, img_shape, embd_size: int):
    h_pad, w_pad = get_pad_sizes_from_img_shape(*img_shape)
    h, w = img_shape
    img_size = max(h, w)
    pad = torch.tensor(
        [w_pad // 2, h_pad // 2],
        device=embd_coords_xy.device,
        dtype=embd_coords_xy.dtype,
    )
    # NOTE +0.5 makes huge difference
    img_coords_xy = ((embd_coords_xy.float() + 0.5) / embd_size * img_size - pad).long()
    # img_coords_xy[:,0] = torch.clamp(img_coords_xy[:,0], 0, w - 1)
    # img_coords_xy[:,1] = torch.clamp(img_coords_xy[:,1], 0, h - 1)
    return img_coords_xy


def zero_pad_square(x: torch.Tensor):
    pad_widths = get_pad_sizes(x)
    return F.pad(
        x[None, None, :, :],
        [
            0,
            0,
            pad_widths[0] // 2,
            pad_widths[0] // 2,
            pad_widths[1] // 2,
            pad_widths[1] // 2,
        ],
        value=0,
    )[0, 0]


def inv_pad_resize_img(img_resized_padded: torch.Tensor, orig_img_h, orig_img_w):
    """
    Args:
        img_resized_padded: (H, W) tensor where H == W and has been padded to be square
    """
    pad_h, pad_w = get_pad_sizes_from_img_shape(orig_img_h, orig_img_w)
    max_size = max(orig_img_h, orig_img_w)
    # upsample img
    img_max = nn.Upsample(size=(max_size, max_size), mode="bilinear")(
        img_resized_padded[None, None, :, :]
    )[0, 0]
    # shape (img_size, img_size)
    # crop img
    img_cropped = img_max[
        pad_h // 2 : pad_h // 2 + orig_img_h, pad_w // 2 : pad_w // 2 + orig_img_w
    ]
    return img_cropped


def inv_pad_resize_attention(attn_resized_padded: torch.Tensor, orig_img_h, orig_img_w):
    """
    Args:
        img_resized_padded: (B, H, W) tensor where H == W and has been padded to be square
    """
    pad_h, pad_w = get_pad_sizes_from_img_shape(orig_img_h, orig_img_w)
    max_size = max(orig_img_h, orig_img_w)
    # upsample img
    upsampler = nn.Upsample(size=(max_size, max_size), mode="bilinear")
    attn_max = upsampler(attn_resized_padded[None, :, :, :])[
        0
    ]  # shape (b, img_size, img_size)

    # crop img
    attn_cropped = attn_max[
        :, pad_h // 2 : pad_h // 2 + orig_img_h, pad_w // 2 : pad_w // 2 + orig_img_w
    ]
    return attn_cropped


def inv_pad_resize_img_rgb(img_resized_padded: torch.Tensor, orig_img_h, orig_img_w):
    """
    Args:
        img_resized_padded: (H, W, 3) tensor where H == W and has been padded to be square
    """
    img_resized_padded = img_resized_padded.permute(2, 0, 1)  # change to (C, H, W)
    pad_h, pad_w = get_pad_sizes_from_img_shape(orig_img_h, orig_img_w)
    max_size = max(orig_img_h, orig_img_w)
    # upsample img
    img_max = nn.Upsample(size=(max_size, max_size), mode="bilinear")(
        img_resized_padded[None, :, :, :]
    )[
        0
    ]  # shape (img_size, img_size)
    # crop img
    img_cropped = img_max[
        :, pad_h // 2 : pad_h // 2 + orig_img_h, pad_w // 2 : pad_w // 2 + orig_img_w
    ].permute(
        1, 2, 0
    )  # change back to (H, W, C)
    return img_cropped


def pad_img_rgb(img: torch.Tensor):
    pad_h, pad_w = get_pad_sizes_from_img_shape(img.shape[0], img.shape[1])
    # pad img to square
    img = img.permute(2, 0, 1)  # change to (C, H, W)
    img_padded = F.pad(
        img[None, :, :, :],
        [0, 0, pad_h // 2, pad_h // 2, pad_w // 2, pad_w // 2],
        value=0,
    )[0]
    return img_padded.permute(1, 2, 0)  # change back to (H, W, C)


# ==========================================
# Old functions for backward compatibility
# TODO remove or refactor for better speed
# ==========================================


def rot(x, y, s, angle_deg):
    angle_rad = np.radians(angle_deg)
    x_c = x - s // 2
    y_c = y - s // 2
    x_new = x_c * np.cos(angle_rad) - y_c * np.sin(angle_rad)
    y_new = x_c * np.sin(angle_rad) + y_c * np.cos(angle_rad)
    x_new += s // 2
    y_new += s // 2
    return x_new, y_new


def transform_point(x, y, s, flip=0, angle_deg=0):
    if flip == 1:
        x = s - 1 - x
    if angle_deg:
        x, y = rot(x, y, s, -angle_deg)
    return x, y


def transform_point_inv(x, y, s, flip=0, angle_deg=0):
    if angle_deg:
        x, y = rot(x, y, s, angle_deg)
    if flip == 1:
        x = s - x
    return x, y


def img_to_embedding_coords(x, y, img_w=768, embd_w=48, img_h=None, embd_h=None):
    """
    Args:
        x (int): x coordinate in image
        y (int): y coordinate in image
        img_w (int): width of image
        embd_w (int): width of embedding
        img_h (int): height of image
        embd_h (int): height of embedding
    """
    embd_h = embd_h or embd_w
    img_h = img_h or img_w
    x = x * embd_w // img_w
    y = y * embd_h // img_h
    return int(x), int(y)


def embd_to_img_coords(x_embd, y_embd, img_w=768, embd_w=48, img_h=None, embd_h=None):
    """
    Args:
        x_embd (int): x coordinate in embedding
        y_embd (int): y coordinate in embedding
        img_w (int): width of image
        embd_w (int): width of embedding
        img_h (int): height of image
        embd_h (int): height of embedding
    """
    embd_h = embd_h or embd_w
    img_h = img_h or img_w
    x = x_embd * img_w // embd_w
    y = y_embd * img_h // embd_h
    return int(x), int(y)


def transform_image_coords(
    x_orig: int,
    y_orig: int,
    img_orig_width: int,
    img_orig_height: int,
    img_new_size: int,
    pad=True,
):
    """Transforms image coordinates to new image size. (new image is square)
    Args:
        x_orig: x coordinate in original image
        y_orig: y coordinate in original image
        img_orig_width: width of original image
        img_orig_height: height of original image
        img_new_size: size of new image
        pad: whether to pad the original image to make it square
    """
    # (optional) pad orig image to square image
    img_orig_height_pad, img_orig_width_pad = img_orig_height, img_orig_width
    pad_x_half, pad_y_half = 0, 0
    if pad:
        if img_orig_height < img_orig_width:
            pad_y_half = np.floor((img_orig_width - img_orig_height) / 2)
        elif img_orig_width < img_orig_height:
            pad_x_half = np.floor((img_orig_height - img_orig_width) / 2)
        max_orig_size = max(img_orig_width, img_orig_height)
        img_orig_height_pad = max_orig_size
        img_orig_width_pad = max_orig_size
    x_orig_pad = x_orig + pad_x_half
    y_orig_pad = y_orig + pad_y_half
    # first normalize and then resize to new image size
    x_new = int(np.floor(x_orig_pad / img_orig_width_pad * img_new_size))
    y_new = int(np.floor(y_orig_pad / img_orig_height_pad * img_new_size))
    return x_new, y_new


def transform_image_coords_parallel(
    img_coords: Union[torch.Tensor, np.ndarray],
    img_orig_width: int,
    img_orig_height: int,
    img_new_size: int,
):
    """
    Args:
        img_coords: (n, 2)
        img_orig_width: int
        img_orig_height: int
        img_new_size: int
    """
    embd_coords_xy = []
    for i in range(len(img_coords)):
        x_new, y_new = transform_image_coords(
            x_orig=int(img_coords[i, 0]),
            y_orig=int(img_coords[i, 1]),
            img_orig_width=img_orig_width,
            img_orig_height=img_orig_height,
            img_new_size=img_new_size,
            pad=True,
        )
        embd_coords_xy.append([x_new, y_new])
    embd_coords_xy_torch = torch.tensor(embd_coords_xy)
    return embd_coords_xy_torch


def transform_image_coords_inv(
    x_new, y_new, img_orig_width, img_orig_height, img_new_size, pad=True
):
    img_orig_height_pad, img_orig_width_pad = img_orig_height, img_orig_width
    pad_x_half, pad_y_half = 0, 0
    if pad:
        if img_orig_height < img_orig_width:
            pad_y_half = np.floor((img_orig_width - img_orig_height) / 2)
        elif img_orig_width < img_orig_height:
            pad_x_half = np.floor((img_orig_height - img_orig_width) / 2)
        max_orig_size = max(img_orig_width, img_orig_height)
        img_orig_height_pad = max_orig_size
        img_orig_width_pad = max_orig_size
    x_orig_pad = int(np.floor(x_new / img_new_size * img_orig_width_pad))
    y_orig_pad = int(np.floor(y_new / img_new_size * img_orig_height_pad))
    x_orig = x_orig_pad - pad_x_half
    y_orig = y_orig_pad - pad_y_half
    return x_orig, y_orig
