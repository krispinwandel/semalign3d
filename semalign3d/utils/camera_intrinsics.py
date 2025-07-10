from typing import Union
import torch
import numpy as np


def get_normalized_calibration_matrix_np(img_h, img_w, f=0.3):
    px = img_w / 2 / max(img_w, img_h)
    py = img_h / 2 / max(img_w, img_h)
    K = np.array([[f, 0, px], [0, f, py], [0, 0, 1]])
    return K


def get_normalized_calibration_matrix_torch(
    img_h: torch.Tensor,
    img_w: torch.Tensor,
    f: Union[torch.Tensor, float],
):
    if isinstance(f, float):
        f = torch.tensor(f, device="cpu", dtype=torch.float32)
    max_size = max(img_w, img_h)
    px = img_w / 2 / max_size
    py = img_h / 2 / max_size
    K = torch.tensor(
        [[f, 0, px, 0], [0, f, py, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        device=f.device,
        dtype=f.dtype,
    )
    return K


# @torch.compile()
def get_calibration_matrix_torch_batched(
    img_h: torch.Tensor,
    img_w: torch.Tensor,
    f: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
):
    """
    Args:
        img_h: (1,)
        img_w: (1,)
        f: (B,)
    Out:
        K: (B, 4, 4)
    """
    B = f.shape[0]
    px = img_w / 2 / torch.maximum(img_w, img_h)
    py = img_h / 2 / torch.maximum(img_w, img_h)

    K = torch.zeros((B, 3, 3), device=device, dtype=dtype)
    K[:, 0, 0] = f
    K[:, 1, 1] = f
    K[:, 0, 2] = px
    K[:, 1, 2] = py
    K[:, 2, 2] = 1

    return K


def get_calibration_matrix_unnormalized_torch(
    img_h: torch.Tensor,
    img_w: torch.Tensor,
    f: torch.Tensor,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
):
    px = img_w / 2
    py = img_h / 2
    K = torch.tensor(
        [
            [f, 0, px],
            [0, f, py],
            [0, 0, 1],
        ],
        device=device,
        dtype=dtype,
    )
    return K


# def get_calibration_matrix_torch_batched(
#     img_h: torch.Tensor,
#     img_w: torch.Tensor,
#     f: torch.Tensor,
#     device: torch.device,
#     dtype: torch.dtype,
# ):
#     """
#     Args:
#         img_h: (1,)
#         img_w: (1,)
#         f: (B,)
#     Out:
#         K: (B, 4, 4)
#     """
#     B = f.shape[0]
#     px = img_w / 2 / torch.maximum(img_w, img_h)
#     py = img_h / 2 / torch.maximum(img_w, img_h)

#     K = torch.zeros((B, 4, 4), device=device, dtype=dtype)
#     K[:, 0, 0] = f
#     K[:, 1, 1] = f
#     K[:, 0, 2] = px
#     K[:, 1, 2] = py
#     K[:, 2, 2] = 1
#     K[:, 3, 3] = 1

#     return K


def get_embd_calibration_matrix_torch_batched(
    img_h: torch.Tensor,
    img_w: torch.Tensor,
    s_embd,
    f: torch.Tensor,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
):
    """
    Args:
        img_h: (1,)
        img_w: (1,)
        s_embd: (1,)
        f: (B,)
    Out:
        K: (B, 4, 4)
    """
    B = f.shape[0]
    s_max = torch.maximum(img_w, img_h)
    pad_x, pad_y = s_max - img_w, s_max - img_h
    px = img_w / 2
    py = img_h / 2

    K = torch.zeros((B, 4, 4), device=device, dtype=dtype)
    K[:, 0, 0] = f * s_embd
    K[:, 1, 1] = f * s_embd
    K[:, 0, 2] = (px + pad_x / 2) * s_embd / s_max
    K[:, 1, 2] = (py + pad_y / 2) * s_embd / s_max
    K[:, 2, 2] = 1
    K[:, 3, 3] = 1

    return K


def get_inv_calibration_matrix(img_h, img_w, f):
    """Construct inverse of calibration matrix analytically"""
    px = img_w / 2 / max(img_w, img_h)
    py = img_h / 2 / max(img_w, img_h)
    f_plus = f + 1e-8
    K_inv = torch.Tensor(
        [[1 / f_plus, 0, -px / f_plus], [0, 1 / f_plus, -py / f_plus], [0, 0, 1]]
    )
    return K_inv


def invert_intrinsics_batch(K: torch.Tensor):
    """
    Args:
        K: (b, 3, 3)
    Out:
        K_inv: (b, 3, 3)
    """
    K_inv = torch.zeros_like(K)
    K_inv[:, 0, 0] = 1 / K[:, 0, 0]
    K_inv[:, 1, 1] = 1 / K[:, 1, 1]
    K_inv[:, 0, 2] = -K[:, 0, 2] / K[:, 0, 0]
    K_inv[:, 1, 2] = -K[:, 1, 2] / K[:, 1, 1]
    K_inv[:, 2, 2] = 1
    return K_inv


def scale_intrinsics(
    K: torch.Tensor, orig_size: tuple, new_size: tuple
) -> torch.Tensor:
    """
    Scales a camera intrinsic matrix K from original image size to new image size.

    Args:
        K (torch.Tensor): Camera intrinsic matrix of shape (3, 3)
        orig_size (tuple): (height1, width1)
        new_size (tuple): (height2, width2)

    Returns:
        torch.Tensor: Scaled intrinsic matrix of shape (3, 3)
    """
    H1, W1 = orig_size
    H2, W2 = new_size

    sx = W2 / W1
    sy = H2 / H1

    K_scaled = K.clone()
    K_scaled[0, 0] *= sx  # fx
    K_scaled[1, 1] *= sy  # fy
    K_scaled[0, 2] *= sx  # cx
    K_scaled[1, 2] *= sy  # cy

    return K_scaled


def crop_intrinsics(K: torch.Tensor, crop_bbox_min) -> torch.Tensor:
    """
    Adjust camera intrinsics after cropping the image.

    Args:
        K (torch.Tensor): Original 3x3 camera intrinsic matrix.
        crop_bbox_min (tuple): (x_min, y_min) in pixels.

    Returns:
        torch.Tensor: New 3x3 intrinsic matrix adjusted for the crop.
    """
    x_min, y_min = crop_bbox_min

    K_crop = K.clone()
    K_crop[0, 2] -= x_min  # cx
    K_crop[1, 2] -= y_min  # cy

    return K_crop


def uncrop_intrinsics(K_crop: torch.Tensor, crop_bbox_min: tuple) -> torch.Tensor:
    """
    Given intrinsics for a cropped image, compute intrinsics for the original image.

    Args:
        K_crop (torch.Tensor): 3x3 intrinsic matrix for the cropped image.
        crop_bbox_min (tuple): (x_min, y_min)
    Returns:
        torch.Tensor: 3x3 intrinsic matrix for the original image.
    """
    x_min, y_min = crop_bbox_min

    K_orig = K_crop.clone()
    K_orig[0, 2] += x_min  # cx
    K_orig[1, 2] += y_min  # cy

    return K_orig


def uncrop_depth_and_intrinsics(
    padded_depth, padded_intrinsics, pad_rect, bbox, img_shape, fill_value
):
    """
    Apply the following steps and adjust instrinsics accordingly:
    1) Unpad the depth map using the padding rectangle.
    2) Resize the depth map to the bounding box shape.
    3) Create a depth map D with fill_value and shape img_shape
    4) Set D[bbox] = resized depth map.

    Use case:
    Can help when predicting depth & intrinsics for a cropped image and then
    uncrop depth & intrinsics to the original image shape. For example, use VGGT
    to predict depth and intrinsics in a bounding box of an image, and then use this
    function to reconstruct the original depth map and intrinsics.

    Args:
        padded_depth: Depth map in the bounding box. Shape (h, w)
        padded_intrinsics: (3,3) Intrinsic matrix of the camera.
        pad_rect: Tuple of padding rectangle coordinates (y_min, x_min, y_max, x_max).
        bbox: Bounding box coordinates [x_min, y_min, x_max, y_max].
        img_shape: Shape of the original image (height, width).
    """
    bbox_depth = padded_depth[pad_rect[0] : pad_rect[2], pad_rect[1] : pad_rect[3]]
    bbox_intrinsic = crop_intrinsics(padded_intrinsics, [pad_rect[1], pad_rect[0]])

    bbox_shape = bbox[3] - bbox[1], bbox[2] - bbox[0]

    # resize depth map to bbox shape
    bbox_depth_resized = (
        torch.nn.functional.interpolate(
            bbox_depth.unsqueeze(0).unsqueeze(0),
            size=bbox_shape,
            mode="bilinear",
            align_corners=False,
        )
        .squeeze(0)
        .squeeze(0)
    )
    bbox_depth_resized = bbox_depth_resized.squeeze(0).squeeze(0)

    # construct the original depth map using fill value
    depth_orig = torch.full(
        (img_shape[0], img_shape[1]), fill_value=fill_value, dtype=torch.float32
    )
    depth_orig[bbox[1] : bbox[3], bbox[0] : bbox[2]] = bbox_depth_resized

    # construct the original intrinsic matrix
    # scale intrinsics
    K_scaled = scale_intrinsics(
        bbox_intrinsic, bbox_depth.shape, bbox_depth_resized.shape
    )
    # uncrop intrinsics
    K_orig = uncrop_intrinsics(K_scaled, bbox[:2])

    return depth_orig, K_orig
