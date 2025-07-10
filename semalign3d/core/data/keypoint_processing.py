"""Keypoint processing utilities"""

import torch
import numpy as np
from tqdm import tqdm
from typing import List
from semalign3d.utils import (
    image_processing,
    torch_utils,
    projections,
    camera_intrinsics,
)
from semalign3d.core import data_classes


def query_xy_coords_to_xyz_with_correction(
    query_img_attn, img_xyz_orig, query_xy_coords, thresh_rate=0.8, window_size=10
):
    """Transform xy coordinates to xyz coordinates with *correction*.
    Args:
        query_xy_coords: (2,)
        query_img_attn: (h_embd, w_embd)
        img_xyz_orig: (h_orig, w_orig, 3)
    Returns:
        query_xyz_coords: (3,) - xyz coordinates of the point in the original image
    """
    h_orig, w_orig = img_xyz_orig.shape[:2]
    attn = image_processing.inv_pad_resize_img(query_img_attn, h_orig, w_orig)
    x, y = query_xy_coords
    min_attn_value = thresh_rate * attn[y, x]
    # look around (x,y) and take candidate with lowest depth
    il, ih = max(0, y - window_size), min(h_orig, y + window_size)
    jl, jh = max(0, x - window_size), min(w_orig, x + window_size)
    candidates = img_xyz_orig[il:ih, jl:jh, :][attn[il:ih, jl:jh] > min_attn_value]
    selected_candidate_idx = torch.argmin(candidates[:, 2])
    query_xyz_coords = candidates[selected_candidate_idx]
    return query_xyz_coords


def query_img_coords_to_xyz_with_correction(
    query_img_coords_xy: torch.Tensor,
    query_img_attn: torch.Tensor,
    img_xyz_orig: torch.Tensor,
    thresh_rate=0.8,
    window_size=10,
):
    """Transform query img coords to xyz coords with *correction*
    Note:
        We correct xy coordinates by considering the attention within a window around the query point
        and change it to the coordinate with attention above thresh_rate and minimum depth.
    Args:
        query_img_coords: (N, 2)
        query_img_attn: (q, h_embd, w_embd)
        img_xyz_orig: (h, w, 3)
        thresh_rate: float - threshold rate for attention
        window_size: int - size of the window to consider around the query point
    Returns:
        query_xyz_coords: (N, 3) - xyz coordinates of the queries in the original image
    """
    query_xyz_coords_list = []
    for i in range(query_img_coords_xy.shape[0]):
        query_xyz = query_xy_coords_to_xyz_with_correction(
            query_img_attn=query_img_attn[i, :, :],
            img_xyz_orig=img_xyz_orig,
            query_xy_coords=query_img_coords_xy[i, :],
            thresh_rate=thresh_rate,
            window_size=window_size,
        )
        query_xyz_coords_list.append(query_xyz)
    query_xyz_coords = torch.stack(query_xyz_coords_list, dim=0)
    return query_xyz_coords


def correct_query_coords(img_seg_mask: np.ndarray, query_coords: np.ndarray):
    """
    Note:
        For each point in query coord, check if it is in the segmentation mask.
        If not, find the closest point in the mask and replace the point with it.
    Args:
        img_seg_mask: (H, W) - binary mask where 1 indicates the object region
        query_coords: (N, 2) - query coordinates in the image (x, y)
    Returns:
        query_coords_corrected: (N, 2) - corrected keypoint coordinates
        mask: (H, W) - eroded segmentation mask
    """
    mask = image_processing.erode_mask(img_seg_mask)
    mask_indices = np.argwhere(mask)
    # permute x and y
    mask_indices = mask_indices[:, [1, 0]]

    # Calculate distances using broadcasting
    distances = np.sqrt(
        ((mask_indices[:, None, :] - query_coords[None, :, :]) ** 2).sum(axis=2)
    )  # shape (num_mask_points, num_keypoints)

    # Find the index of the closest point in the mask for each keypoint
    closest_indices = np.argmin(distances, axis=0)
    query_coords_corrected = mask_indices[closest_indices]

    return query_coords_corrected, mask


# ===========================================
# Wrappers around SemAlign3D data structures
# ===========================================


def correct_all_keypoint_coords(
    img_seg_masks: List[np.ndarray],
    kpt_img_coords: List[torch.Tensor],
    img_xyz_orig: List[np.ndarray],
    show_progress_bar=False,
):
    """Correct keypoint coordinates with eroded segmentation masks.
    Args:
        img_seg_masks: List of segmentation mask triplets for each image, each mask is (H, W, 3) - binary mask where 1 indicates the object region
        kpt_img_coords: List of keypoint coordinates for each image, each tensor is (N, 3) - (x, y, kpt_id)
        img_xyz_orig: List of original images with xyz coordinates, each image is (H, W, 3)
        show_progress_bar: bool - whether to show progress bar
    Returns:
        kpt_coords_corrected_normalized_all: (n_imgs, n_max_kpts, 2) - normalized corrected keypoint coordinates
        depth_values_all: (n_imgs, n_max_kpts) - depth values of the keypoints
        kpt_xyz_list: List of tensors with xyz coordinates of the keypoints for each image
        masks: List of masks for each image
        kpt_coords_corrected_all: List of corrected keypoint coordinates for each image
    """
    n_imgs = len(img_xyz_orig)
    n_max_kpts = (
        np.max([torch.max(kpt_img_coords[i][:, 2]).item() for i in range(n_imgs)]) + 1
    )
    kpt_coords_corrected_all = []
    kpt_coords_corrected_normalized_all = torch.full(
        (n_imgs, n_max_kpts, 2), dtype=torch.float32, fill_value=torch.nan
    )
    depth_values_all = torch.full(
        (n_imgs, n_max_kpts), dtype=torch.float32, fill_value=torch.nan
    )
    kpt_xyz_list = []
    masks = []
    for i in tqdm(range(n_imgs), disable=not show_progress_bar):
        kpt_coords = kpt_img_coords[i][:, :2]
        kpt_ids = kpt_img_coords[i][:, 2]
        kpt_coords_corrected, mask = correct_query_coords(
            img_seg_masks[i][2], kpt_coords.numpy()
        )
        kpt_coords_corrected_all.append(kpt_coords_corrected)
        masks.append(mask)
        kpt_coords_corrected_normalized = kpt_coords_corrected / max(
            img_xyz_orig[i].shape[:2]
        )
        kpt_coords_corrected_normalized_all[i, kpt_ids] = torch_utils.to_torch_tensor(
            kpt_coords_corrected_normalized, device="cpu", dtype=torch.float32
        ).float()

        kpt_xyz = img_xyz_orig[i][
            kpt_coords_corrected[:, 1], kpt_coords_corrected[:, 0]
        ]
        kpt_xyz_list.append(kpt_xyz)

        depth_values = kpt_xyz[:, 2]
        depth_values_all[i, kpt_ids] = torch_utils.to_torch_tensor(
            depth_values, device="cpu", dtype=torch.float32
        ).float()

    return (
        kpt_coords_corrected_normalized_all,
        depth_values_all,
        kpt_xyz_list,
        masks,
        kpt_coords_corrected_all,
    )


def correct_all_keypoint_coords_with_depths(
    img_seg_masks: List[np.ndarray],
    kpt_img_coords: List[torch.Tensor],
    depths: List[np.ndarray],
    show_progress_bar=False,
    focal_length=5.0,
):
    img_xyz_list = [
        projections.back_project_depth_image(depth, focal_length=focal_length)
        for depth in depths
    ]
    return correct_all_keypoint_coords(
        img_seg_masks=img_seg_masks,
        kpt_img_coords=kpt_img_coords,
        img_xyz_orig=img_xyz_list,
        show_progress_bar=show_progress_bar,
    )


def correct_gt_kpts_xyz(preprocessed_data_train: data_classes.SpairProcessedData):
    """Correct xy image coords of ground truth keypoints and then back-project them to 3D space."""
    # 1) compute input args for optimization procedure
    (
        gt_kpt_xy_all_normalized,
        depth_values_all,
        gt_kpt_xyz_list,
        masks,
        kpt_coords_corrected_all,
    ) = correct_all_keypoint_coords(
        img_seg_masks=preprocessed_data_train.seg_masks,
        kpt_img_coords=preprocessed_data_train.kpt_img_coords,
        img_xyz_orig=preprocessed_data_train.xyz,
    )

    img_shapes = torch.zeros((gt_kpt_xy_all_normalized.shape[0], 2))
    for i in range(gt_kpt_xy_all_normalized.shape[0]):
        img_shapes[i, :] = torch.tensor(preprocessed_data_train.xyz[i].shape[:2])

    gt_kpt_xy_all_normalized_no_nan = gt_kpt_xy_all_normalized.clone()
    gt_kpt_xy_all_normalized_no_nan[torch.isnan(gt_kpt_xy_all_normalized_no_nan)] = 0.0
    depth_values_all_no_nan = depth_values_all.clone()
    depth_values_all_no_nan[torch.isnan(depth_values_all_no_nan)] = 0.0

    vertex_mask = torch_utils.generate_is_not_nan_mask(gt_kpt_xy_all_normalized)

    # reproject all points with optimised focal length
    # kpts_xyz_all = projection_utils.reproject_kpts(
    #     focal_lenghts_inv=torch.abs(torch_utils.to_torch_tensor(preprocessed_data_train.focal_lengths_inv, device="cpu")),
    #     img_shapes=img_shapes,
    #     kpt_xy_normalized=gt_kpt_xy_all_normalized_no_nan,
    #     kpt_depth=depth_values_all_no_nan,
    # )
    kpts_xyz_all = projections.back_project_kpts_from_K_inv_normalized(
        K_inv_normalized=camera_intrinsics.invert_intrinsics_batch(
            torch.from_numpy(
                preprocessed_data_train.normalized_camera_intrinsics
            ).float()
        ),
        kpt_xy_normalized=gt_kpt_xy_all_normalized_no_nan.float(),
        kpt_depth=depth_values_all_no_nan.float(),
    )

    gt_kpts_data = data_classes.GtKptsData(
        xyz=kpts_xyz_all,
        vertex_mask=vertex_mask,
    )

    return gt_kpts_data
