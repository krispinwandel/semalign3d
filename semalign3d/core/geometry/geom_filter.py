import torch
from typing import List
from semalign3d.core import data_classes
from semalign3d.core.geometry import geom_features
from semalign3d.utils.stats import beta_dist


def generate_angle_and_ratio_mask(kpts_is_not_nan):
    """
    Generate masks for valid angles and ratios based on keypoints.
    """
    pseudo_kpt_xyz = torch.zeros(
        (kpts_is_not_nan.shape[0], kpts_is_not_nan.shape[1], 3)
    )
    pseudo_kpt_xyz[~kpts_is_not_nan, :] = torch.nan
    pseudo_cos_angles, _, pseudo_ratios = (
        geom_features.compute_angles_and_ratios_parallel(pseudo_kpt_xyz)
    )
    angles_mask = ~torch.isnan(pseudo_cos_angles)
    ratios_mask = ~torch.isnan(pseudo_ratios)
    return angles_mask, ratios_mask


def remove_unused_keypoints(
    kpt_img_coords: List[torch.Tensor],
    geom_comb: data_classes.GeomRelationCombinations,
    geom_stats: data_classes.GeomRelationStatisticsBetaSimple,
):
    n_max_kpts = 30  # each category has at most 30 keypoints
    unused_kpts = []
    kpt_counts = torch.zeros(n_max_kpts, dtype=torch.int64)
    for img_idx in range(len(kpt_img_coords)):
        img_kpt_labels = kpt_img_coords[img_idx][:, 2]
        kpt_counts += torch.bincount(img_kpt_labels, minlength=n_max_kpts)
    unused_kpts = torch.where(kpt_counts == 0)[0]
    if len(unused_kpts) > 0:
        e1_filter = ~torch.any(torch.isin(geom_comb.e1, unused_kpts), dim=-1)
        e2_filter = ~torch.any(torch.isin(geom_comb.e2, unused_kpts), dim=-1)
        e_filter = e1_filter & e2_filter
        t_filter = ~torch.any(torch.isin(geom_comb.t, unused_kpts), dim=-1)
        v_filter = ~torch.isin(geom_comb.v, unused_kpts)
        tet_filter = t_filter & v_filter

        geom_comb = data_classes.GeomRelationCombinations(
            e1=geom_comb.e1[e_filter],
            e2=geom_comb.e2[e_filter],
            t=geom_comb.t[tet_filter],
            v=geom_comb.v[tet_filter],
        )
        geom_stats = data_classes.GeomRelationStatisticsBetaSimple(
            edge_pair_cos_angles_alpha=geom_stats.edge_pair_cos_angles_alpha[e_filter],
            edge_pair_cos_angles_beta=geom_stats.edge_pair_cos_angles_beta[e_filter],
            edge_pair_ratios_alpha=geom_stats.edge_pair_ratios_alpha[e_filter],
            edge_pair_ratios_beta=geom_stats.edge_pair_ratios_beta[e_filter],
            cos_angle_x_alpha=geom_stats.cos_angle_x_alpha[tet_filter],
            cos_angle_x_beta=geom_stats.cos_angle_x_beta[tet_filter],
            cos_angle_y_alpha=geom_stats.cos_angle_y_alpha[tet_filter],
            cos_angle_y_beta=geom_stats.cos_angle_y_beta[tet_filter],
            cos_angle_z_alpha=geom_stats.cos_angle_z_alpha[tet_filter],
            cos_angle_z_beta=geom_stats.cos_angle_z_beta[tet_filter],
        )
    n_max_kpts = 1 + torch.max(torch.where(kpt_counts > 0)[0]).item()
    unused_kpts = unused_kpts[unused_kpts < n_max_kpts]
    return geom_comb, geom_stats, unused_kpts, n_max_kpts


def remove_tets_from_geom_relations(
    geom_relation_combinations: data_classes.GeomRelationCombinations,
    geom_relation_stats: data_classes.GeomRelationStatisticsBetaSimple,
):
    # 1) only keep triangle combinations instead of ijkl => ijk
    # 2) only keep edge angle and ratio
    m1 = geom_relation_combinations.e1[:, 0] == geom_relation_combinations.e2[:, 0]
    return {
        "geom_relation_combinations": data_classes.GeomRelationCombinations(
            e1=geom_relation_combinations.e1[m1],
            e2=geom_relation_combinations.e2[m1],
            t=torch.tensor([]),
            v=torch.tensor([]),
        ),
        "geom_relation_stats": data_classes.GeomRelationStatisticsBetaSimple(
            edge_pair_cos_angles_alpha=geom_relation_stats.edge_pair_cos_angles_alpha[
                m1
            ],
            edge_pair_cos_angles_beta=geom_relation_stats.edge_pair_cos_angles_beta[m1],
            edge_pair_ratios_alpha=geom_relation_stats.edge_pair_ratios_alpha[m1],
            edge_pair_ratios_beta=geom_relation_stats.edge_pair_ratios_beta[m1],
            cos_angle_x_alpha=torch.tensor([]),
            cos_angle_x_beta=torch.tensor([]),
            cos_angle_y_alpha=torch.tensor([]),
            cos_angle_y_beta=torch.tensor([]),
            cos_angle_z_alpha=torch.tensor([]),
            cos_angle_z_beta=torch.tensor([]),
        ),
    }


def filter_geom_relations_chosen_connected_with_next(
    next_kpt_index: int,
    chosen_kpt_indices: List[int],
    geom_stats_beta: data_classes.GeomRelationStatisticsBetaSimple,
    geom_relation_combinations: data_classes.GeomRelationCombinations,
    n_max_comb=100,
):
    """Filter geometric relations based on chosen keypoints and next keypoint index."""
    # filter geom combinations
    n_e = geom_relation_combinations.e1.shape[0]
    n_tet = geom_relation_combinations.t.shape[0]
    e_sum = torch.zeros(n_e, dtype=torch.int16)
    tet_sum = torch.zeros(n_tet, dtype=torch.int16)
    for kpt_index in chosen_kpt_indices:
        e_sum += (
            torch.any(geom_relation_combinations.e1 == kpt_index, dim=-1).int()
            + torch.any(geom_relation_combinations.e2 == kpt_index, dim=-1).int()
        )
        tet_sum += (
            torch.any(geom_relation_combinations.t == kpt_index, dim=-1).int()
            + (geom_relation_combinations.v == kpt_index).int()
        )
    e_sum_must = (
        torch.any(geom_relation_combinations.e1 == next_kpt_index, dim=-1).int()
        + torch.any(geom_relation_combinations.e2 == next_kpt_index, dim=-1).int()
    )
    tet_sum_must = (
        torch.any(geom_relation_combinations.t == next_kpt_index, dim=-1).int()
        + (geom_relation_combinations.v == next_kpt_index).int()
    )
    e_filter = (e_sum >= 3) & (e_sum_must >= 1)
    tet_filter = (tet_sum >= 3) & (tet_sum_must >= 1)
    # geom_stats_var = geom_relations.compute_beta_var(geom_stats_filtered)
    # n_max_comb = 100
    # TODO use combinations with low var
    # geom_stats_var = geom_relations.compute_beta_var(geom_stats_filtered)
    geom_stats_filtered = data_classes.GeomRelationStatisticsBetaSimple(
        edge_pair_ratios_alpha=geom_stats_beta.edge_pair_ratios_alpha[e_filter][
            :n_max_comb
        ],
        edge_pair_ratios_beta=geom_stats_beta.edge_pair_ratios_beta[e_filter][
            :n_max_comb
        ],
        edge_pair_cos_angles_alpha=geom_stats_beta.edge_pair_cos_angles_alpha[e_filter][
            :n_max_comb
        ],
        edge_pair_cos_angles_beta=geom_stats_beta.edge_pair_cos_angles_beta[e_filter][
            :n_max_comb
        ],
        cos_angle_x_alpha=geom_stats_beta.cos_angle_x_alpha[tet_filter][:n_max_comb],
        cos_angle_x_beta=geom_stats_beta.cos_angle_x_beta[tet_filter][:n_max_comb],
        cos_angle_y_alpha=geom_stats_beta.cos_angle_y_alpha[tet_filter][:n_max_comb],
        cos_angle_y_beta=geom_stats_beta.cos_angle_y_beta[tet_filter][:n_max_comb],
        cos_angle_z_alpha=geom_stats_beta.cos_angle_z_alpha[tet_filter][:n_max_comb],
        cos_angle_z_beta=geom_stats_beta.cos_angle_z_beta[tet_filter][:n_max_comb],
    )
    geom_combinations_filtered = data_classes.GeomRelationCombinations(
        e1=geom_relation_combinations.e1[e_filter][:n_max_comb],
        e2=geom_relation_combinations.e2[e_filter][:n_max_comb],
        t=geom_relation_combinations.t[tet_filter][:n_max_comb],
        v=geom_relation_combinations.v[tet_filter][:n_max_comb],
    )
    return geom_stats_filtered, geom_combinations_filtered


def filter_valid_stats(
    geom_stats: data_classes.GeomRelationStatisticsBeta,
    geom_relation_combinations: data_classes.GeomRelationCombinations,
    n_samples_per_img: int = 10,
    min_sample_imgs=4,  # 4 seems ok
    min_edge_ratio=0.01,
    max_edge_ratio=0.99,
):
    min_samples = min_sample_imgs * n_samples_per_img
    # TODO perhaps we should weight by samples
    e_filter = geom_stats.m_edge_pair_cos_angles.sum(dim=0) >= min_samples
    tet_filter = geom_stats.m_cos_angle.sum(dim=0) >= min_samples
    edge_pair_ratios = beta_dist.compute_beta_mode_torch(
        geom_stats.edge_pair_ratios_alpha, geom_stats.edge_pair_ratios_beta
    )
    e_filter = (
        e_filter
        & (edge_pair_ratios >= min_edge_ratio)
        & (edge_pair_ratios <= max_edge_ratio)
    )
    # make sure alpha and beta are both greater than 1
    e_filter = (
        e_filter
        & (geom_stats.edge_pair_ratios_alpha > 1)
        & (geom_stats.edge_pair_ratios_beta > 1)
        & (geom_stats.edge_pair_cos_angles_alpha > 1)
        & (geom_stats.edge_pair_cos_angles_beta > 1)
    )
    tet_filter = (
        tet_filter
        & (geom_stats.cos_angle_x_alpha > 1)
        & (geom_stats.cos_angle_x_beta > 1)
        & (geom_stats.cos_angle_y_alpha > 1)
        & (geom_stats.cos_angle_y_beta > 1)
        & (geom_stats.cos_angle_z_alpha > 1)
        & (geom_stats.cos_angle_z_beta > 1)
    )
    geom_stats_filtered = data_classes.GeomRelationStatisticsBetaSimple(
        edge_pair_ratios_alpha=geom_stats.edge_pair_ratios_alpha[e_filter],
        edge_pair_ratios_beta=geom_stats.edge_pair_ratios_beta[e_filter],
        edge_pair_cos_angles_alpha=geom_stats.edge_pair_cos_angles_alpha[e_filter],
        edge_pair_cos_angles_beta=geom_stats.edge_pair_cos_angles_beta[e_filter],
        cos_angle_x_alpha=geom_stats.cos_angle_x_alpha[tet_filter],
        cos_angle_x_beta=geom_stats.cos_angle_x_beta[tet_filter],
        cos_angle_y_alpha=geom_stats.cos_angle_y_alpha[tet_filter],
        cos_angle_y_beta=geom_stats.cos_angle_y_beta[tet_filter],
        cos_angle_z_alpha=geom_stats.cos_angle_z_alpha[tet_filter],
        cos_angle_z_beta=geom_stats.cos_angle_z_beta[tet_filter],
    )
    geom_combinations_filtered = data_classes.GeomRelationCombinations(
        e1=geom_relation_combinations.e1[e_filter],
        e2=geom_relation_combinations.e2[e_filter],
        t=geom_relation_combinations.t[tet_filter],
        v=geom_relation_combinations.v[tet_filter],
    )
    return geom_stats_filtered, geom_combinations_filtered
