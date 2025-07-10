from typing import Dict
import torch

from semalign3d.core.losses import geom_loss
from semalign3d.utils import projections


def calculate_loss_depth_sparse_pc(
    kpt_xyz_obj: torch.Tensor,
    kpt_weights: torch.Tensor,
    depth_mean_target: torch.Tensor,
):
    depth_mean = torch.sum(kpt_xyz_obj[:, :, 2] * kpt_weights.unsqueeze(0), dim=1)
    loss_depth_mean = (depth_mean - depth_mean_target) ** 2  # (B,)
    return loss_depth_mean


def calculate_gt_alignment_loss(
    # focal_length_inv: torch.Tensor,  # (B,)
    k_inv: torch.Tensor,  # (B, 3, 3)
    # img_shape: torch.Tensor,  # (B, 2)
    gt_kpt_xy_normalized: torch.Tensor,  # (n_kpts, 2)
    gt_kpt_labels: torch.Tensor,  # (n_kpts,)
    kpt_depths: torch.Tensor,  # (B, n_kpts)
    kpt_xyz_obj: torch.Tensor,  # (B, n_kpts, 3)
) -> torch.Tensor:
    # sparse_pc_xyz = projections.back_project_kpts(
    #     focal_lenghts_inv=torch.abs(focal_length_inv),
    #     img_shapes=img_shape.unsqueeze(0),
    #     kpt_xy_normalized=gt_kpt_xy_normalized.unsqueeze(0),
    #     kpt_depth=kpt_depths[:, :],
    # )  # (B, n_kpt_ann, 3), n_kpt_ann = number of annotated keypoints

    sparse_pc_xyz = projections.back_project_kpts_from_K_inv_normalized(
        K_inv_normalized=k_inv,
        kpt_xy_normalized=gt_kpt_xy_normalized.unsqueeze(0),
        kpt_depth=kpt_depths[:, :],
    )  # (B, n_kpt_ann, 3), n_kpt_ann = number of annotated keypoints

    abs_dist = torch.abs(
        (kpt_xyz_obj[:, gt_kpt_labels, :] - sparse_pc_xyz[:, :, :])
    )  # (B, n_kpt_ann, 3)
    abs_dist = torch.sum(abs_dist, dim=-1)  # (B, K)
    return torch.mean(abs_dist, dim=-1)


def calculate_keypoint_depth_loss(
    # keypoint props
    kpt_xyz_obj: torch.Tensor,  # to be optimised
    kpt_depths: torch.Tensor,  # to be optimised
    kpt_weights: torch.Tensor,
    gt_kpt_xy_normalized: torch.Tensor,
    gt_kpt_labels: torch.Tensor,
    # img props
    # img_shape: torch.Tensor,
    # img_focal_length_inv: torch.Tensor,
    k_inv: torch.Tensor,  # (B, 3, 3)
    # depth target
    depth_mean_target: torch.Tensor,
    # geom stats
    geom_stats: Dict[str, torch.Tensor],
    geom_relation_combinations: Dict[str, torch.Tensor],
    # param
    w_gt_align: float,
    w_geom: float,
    w_depth_sparse_pc: float,
):

    loss_depth_sparse_pc = calculate_loss_depth_sparse_pc(
        kpt_xyz_obj=kpt_xyz_obj,
        kpt_weights=kpt_weights,
        depth_mean_target=depth_mean_target,
    )

    loss_gt_align = calculate_gt_alignment_loss(
        kpt_xyz_obj=kpt_xyz_obj,  # (B, n_kpts, 3)
        kpt_depths=kpt_depths,  # (B, n_kpts)
        gt_kpt_xy_normalized=gt_kpt_xy_normalized,  # (n_kpts, 2)
        gt_kpt_labels=gt_kpt_labels,  # (n_kpts,)
        k_inv=k_inv,  # (B, 3, 3)
        # focal_length_inv=img_focal_length_inv,  # (B,)
        # img_shape=img_shape,  # (B, 2)
    )

    loss_geom = geom_loss.calculate_geom_loss(
        kpt_xyz_obj=kpt_xyz_obj,
        geom_stats=geom_stats,
        geom_relation_combinations=geom_relation_combinations,
    )

    loss_batched = (
        w_gt_align * loss_gt_align
        + w_geom * loss_geom
        + w_depth_sparse_pc * loss_depth_sparse_pc
    )
    loss = torch.sum(loss_batched, dim=-1)  # (B,)
    return {"loss": loss, "batch_loss": loss_batched}
