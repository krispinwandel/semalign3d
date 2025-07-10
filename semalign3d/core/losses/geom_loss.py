from typing import Dict
import torch
from semalign3d.core.geometry import geom_features
from semalign3d.utils.stats import beta_dist


def calculate_geom_loss_for_edge_pair_ratios(
    kpt_xyz_obj: torch.Tensor,
    geom_stats: Dict[str, torch.Tensor],
    geom_relation_combinations: Dict[str, torch.Tensor],
):
    geom_fts_edge_pairs = geom_features.compute_geom_features_for_edge_pairs(
        vertices=kpt_xyz_obj, geom_relation_combinations=geom_relation_combinations
    )
    t0d = torch.tensor([0.0], device=kpt_xyz_obj.device)  # [0,0] - kpt_xyz_obj[0,0]
    t1d = torch.ones_like(t0d)
    t1negd = -t1d

    edge_pair_ratio_log_prob = beta_dist.compute_log_prob_beta_fast(
        x=geom_fts_edge_pairs["edge_pair_ratios"],
        alpha=geom_stats["edge_pair_ratios_alpha"],
        beta=geom_stats["edge_pair_ratios_beta"],
        x_low=t0d,
        x_high=t1d,
    )
    edge_pair_cos_angle_log_prob = beta_dist.compute_log_prob_beta_fast(
        x=geom_fts_edge_pairs["edge_pair_cos_angles"],
        alpha=geom_stats["edge_pair_cos_angles_alpha"],
        beta=geom_stats["edge_pair_cos_angles_beta"],
        x_low=t1negd,
        x_high=t1d,
    )
    l1 = -torch.mean(edge_pair_ratio_log_prob, dim=-1)
    l2 = -torch.mean(edge_pair_cos_angle_log_prob, dim=-1)
    loss = l1 + l2
    return loss


def calculate_geom_loss_for_cos_angles(
    kpt_xyz_obj: torch.Tensor,
    geom_stats: Dict[str, torch.Tensor],
    geom_relation_combinations: Dict[str, torch.Tensor],
):
    geom_fts_tets = geom_features.compute_geom_features_for_tets(
        vertices=kpt_xyz_obj, geom_relation_combinations=geom_relation_combinations
    )
    t1negd = torch.tensor([-1.0], device=kpt_xyz_obj.device)
    t1d = torch.ones_like(t1negd)
    cos_angle_x_log_prob = beta_dist.compute_log_prob_beta_fast(
        x=geom_fts_tets["cos_angle_x"],
        alpha=geom_stats["cos_angle_x_alpha"],
        beta=geom_stats["cos_angle_x_beta"],
        x_low=t1negd,
        x_high=t1d,
    )
    cos_angle_y_log_prob = beta_dist.compute_log_prob_beta_fast(
        x=geom_fts_tets["cos_angle_y"],
        alpha=geom_stats["cos_angle_y_alpha"],
        beta=geom_stats["cos_angle_y_beta"],
        x_low=t1negd,
        x_high=t1d,
    )
    cos_angle_z_log_prob = beta_dist.compute_log_prob_beta_fast(
        x=geom_fts_tets["cos_angle_z"],
        alpha=geom_stats["cos_angle_z_alpha"],
        beta=geom_stats["cos_angle_z_beta"],
        x_low=t1negd,
        x_high=t1d,
    )
    l1 = -torch.mean(cos_angle_x_log_prob, dim=-1)
    l2 = -torch.mean(cos_angle_y_log_prob, dim=-1)
    l3 = -torch.mean(cos_angle_z_log_prob, dim=-1)
    loss = l1 + l2 + l3
    return loss


@torch.compile
def calculate_geom_loss(
    kpt_xyz_obj: torch.Tensor,
    geom_stats: Dict[str, torch.Tensor],
    geom_relation_combinations: Dict[str, torch.Tensor],
    use_tets: bool = True,
):
    l_edge_pairs = calculate_geom_loss_for_edge_pair_ratios(
        kpt_xyz_obj=kpt_xyz_obj,
        geom_stats=geom_stats,
        geom_relation_combinations=geom_relation_combinations,
    )
    if use_tets:
        l_cos_angles = calculate_geom_loss_for_cos_angles(
            kpt_xyz_obj=kpt_xyz_obj,
            geom_stats=geom_stats,
            geom_relation_combinations=geom_relation_combinations,
        )
    else:
        l_cos_angles = torch.zeros_like(l_edge_pairs)
    loss_geom = (l_edge_pairs + l_cos_angles) / 5.0
    return loss_geom
