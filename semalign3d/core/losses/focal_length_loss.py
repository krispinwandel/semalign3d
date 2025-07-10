import torch
from semalign3d.utils import projections, torch_utils
from semalign3d.core.geometry import geom_features


def compute_angle_ratio_var_mean(angles, ratios, angles_mask, ratios_mask):
    """
    Compute mean and variance of angles and ratios over images.
    """
    angles_var, angles_mean = torch_utils.masked_var_mean(angles, angles_mask)
    ratios_var, ratios_mean = torch_utils.masked_var_mean(ratios, ratios_mask)
    return angles_mean, angles_var, ratios_mean, ratios_var


def calculate_focal_length_loss(
    focal_lengths_inv: torch.Tensor,
    img_shapes: torch.Tensor,
    kpt_xy_normalized: torch.Tensor,
    kpt_depth: torch.Tensor,
    angles_mask: torch.Tensor,
    ratios_mask: torch.Tensor,
):
    """
    Args:
        focal_lenghts_inv: (n_imgs,) 1 / focal_length
        img_shapes: (n_imgs, 2). img_shapes[:,0] = img height (y), img_shapes[:,1] = img width (x)
        kpt_xy_normalized: (n_imgs, n_max_kpts, 2)
        kpt_depth: (n_imgs, n_max_kpts)
    """

    # 1) reproject keypoints to world coordinates
    kpt_xyz = projections.back_project_kpts(
        focal_lenghts_inv=torch.abs(focal_lengths_inv),
        img_shapes=img_shapes,
        kpt_xy_normalized=kpt_xy_normalized,
        kpt_depth=kpt_depth,
    )

    # 2) compute angles and ratios between all 3-tuples of keypoints for each image in parallel
    cos_angles, angles, ratios = geom_features.compute_angles_and_ratios_parallel(
        kpt_xyz
    )
    cos_angles_mean, cos_angles_var, ratios_mean, ratios_var = (
        compute_angle_ratio_var_mean(cos_angles, ratios, angles_mask, ratios_mask)
    )

    # 3) compute energy
    loss_focal = torch.sum(cos_angles_var) + torch.sum(ratios_var)

    return {
        "loss": loss_focal,
        "cos_angles": cos_angles,
        "ratios": ratios,
        "cos_angles_mean": cos_angles_mean,
        "cos_angles_var": cos_angles_var,
        "ratios_mean": ratios_mean,
        "ratios_var": ratios_var,
    }
