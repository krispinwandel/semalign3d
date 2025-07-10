import torch
from tqdm import tqdm
import os
from typing import List

from semalign3d.core import data_classes
from semalign3d.core.geometry import geom_combinations, geom_features
from semalign3d.core.data import (
    augmentations,
    sem3d_data_utils,
    keypoint_processing,
)
from semalign3d.utils.stats import beta_dist
from semalign3d.core.generators.generator import Generator


class GeomBetaDistGenerator(Generator):
    """Class to generate beta distributions for geometric features"""

    def __init__(
        self,
        paths: data_classes.SemAlign3DPaths,
        opt_flag: str = "noopt",
        max_combinations=10000,
        # ground truth keypoint augmentation parameters
        n_samples_per_img=10,
        noise_rate=0.2,
        n_max_kpts=30,
        use_vggt=False,
    ):
        if opt_flag not in ["opt", "noopt"]:
            raise ValueError("Invalid flag. Use 'opt' or 'noopt'.")
        use_optimised_depth = opt_flag == "opt"

        self.paths = paths
        self.geom_stats_suffix = f"_{opt_flag}" if opt_flag in ["opt"] else ""
        self.use_vggt = use_vggt
        self.use_optimised_depth = use_optimised_depth
        self.full = False
        self.max_combinations = max_combinations
        self.n_samples_per_img = n_samples_per_img
        self.noise_rate = noise_rate
        self.n_max_kpts = n_max_kpts

        self.out_dir = paths.geom_stats_dir
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)

    def load_gt_kpts_data(
        self,
        category: str,
    ):
        sem_align_3d_data = sem3d_data_utils.load_data(
            category=category,
            paths=self.paths,
            load_geom_stats=False,
            do_load_sparse_pc=False,
            do_load_dense_pc=False,
            use_vggt=self.use_vggt,
        )
        if not self.use_optimised_depth:
            gt_kpts_data = keypoint_processing.correct_gt_kpts_xyz(
                sem_align_3d_data.full_data.processed_data_train
            )
        else:
            gt_kpts_data = sem3d_data_utils.load_gt_kpts_data_with_optimised_depth(
                processed_data=sem_align_3d_data.full_data.processed_data_train,
                base_dir=f"{self.paths.kpt_optimised_depth_dir}/{category}",
                n_verts_max=self.n_max_kpts,
                use_obj_xyz=False,
                suffix=self.paths.suffix,
            )
        return gt_kpts_data

    def save_beta_stats(
        self,
        category: str,
        geom_stats: data_classes.GeomRelationStatisticsBeta,
        geom_relation_combinations: data_classes.GeomRelationCombinations,
    ):
        torch.save(
            geom_stats,
            f"{self.out_dir}/{category}_geom_stats{self.geom_stats_suffix}{self.paths.suffix}.pt",
        )
        torch.save(
            geom_relation_combinations,
            f"{self.out_dir}/{category}_geom_combinations{self.geom_stats_suffix}{self.paths.suffix}.pt",
        )

    def generate(self, category: str):
        print(f"Processing category: {category}")
        print("Loading data...")
        gt_kpts_data = self.load_gt_kpts_data(category)
        gt_kpts_data = augmentations.generate_noisy_gt_kpts_data(
            gt_kpts_data,
            n_samples_per_img=self.n_samples_per_img,
            noise_rate=self.noise_rate,
        )
        print("Fit beta distributions on geometric features...")
        geom_stats, geom_combinations = fit_beta_on_geom_features(
            gt_kpts_data,
            full=self.full,
            max_combinations=self.max_combinations,
        )
        print("Saving results...")
        self.save_beta_stats(
            category,
            geom_stats,
            geom_combinations,
        )


def fit_beta_on_geom_features(
    gt_kpts_data: data_classes.GtKptsData, full=False, max_combinations=10000
):
    """Compute statistics for geometric relations using beta distributions"""
    # Compute statistics for pseudo data in order to optain masks
    # NOTE xyz of invalid points are set to zero
    n_verts = int(gt_kpts_data.xyz.shape[1])
    pseudo_kpt_xyz = torch.zeros((gt_kpts_data.vertex_mask.shape[0], n_verts, 3))
    pseudo_kpt_xyz[~gt_kpts_data.vertex_mask, :] = torch.nan
    geom_relation_combinations = (
        geom_combinations.generate_geometric_relation_combinations(n_verts, full=full)
    )

    # create data containers
    # NOTE we initialize with ones (uniform distribution) in case fitting fails
    n_edge_pair = geom_relation_combinations.e1.shape[0]
    edge_pair_cos_angles_alpha = torch.ones(n_edge_pair)
    edge_pair_cos_angles_beta = torch.ones(n_edge_pair)
    edge_pair_ratios_alpha = torch.ones(n_edge_pair)
    edge_pair_ratios_beta = torch.ones(n_edge_pair)

    n_tets = geom_relation_combinations.t.shape[0]
    cos_angle_x_alpha = torch.ones(n_tets)
    cos_angle_x_beta = torch.ones(n_tets)
    cos_angle_y_alpha = torch.ones(n_tets)
    cos_angle_y_beta = torch.ones(n_tets)
    cos_angle_z_alpha = torch.ones(n_tets)
    cos_angle_z_beta = torch.ones(n_tets)

    n_samples = gt_kpts_data.xyz.shape[0]
    m_edge_pair_cos_angles = torch.zeros((n_samples, n_edge_pair), dtype=torch.bool)
    m_edge_pair_ratios = torch.zeros((n_samples, n_edge_pair), dtype=torch.bool)
    m_cos_angle = torch.zeros((n_samples, n_tets), dtype=torch.bool)

    # process data in batches
    for c in tqdm(range(0, n_edge_pair, max_combinations)):
        geom_relation_combinations_batch = data_classes.GeomRelationCombinations(
            e1=geom_relation_combinations.e1[c : c + max_combinations],
            e2=geom_relation_combinations.e2[c : c + max_combinations],
            t=geom_relation_combinations.t[:1],
            v=geom_relation_combinations.v[:1],
        )

        pseudo_geom_fts = geom_features.compute_geom_features_batched(
            vertices=pseudo_kpt_xyz,
            geom_relation_combinations=geom_relation_combinations_batch,
        )
        m_edge_pair_cos_angles[:, c : c + max_combinations] = ~torch.isnan(
            pseudo_geom_fts.edge_pair_cos_angles
        )
        m_edge_pair_ratios[:, c : c + max_combinations] = ~torch.isnan(
            pseudo_geom_fts.edge_pair_ratios
        )

        # Compute geom_relations of gt_kpts_data
        geom_fts = geom_features.compute_geom_features_batched(
            vertices=gt_kpts_data.xyz,
            geom_relation_combinations=geom_relation_combinations_batch,
        )

        # Fit beta distributions on geom_relations
        for j in range(min(max_combinations, n_edge_pair - c)):
            i = c + j
            if torch.sum(m_edge_pair_cos_angles[:, i]) == 0:
                continue
            try:
                edge_pair_cos_angles_alpha[i], edge_pair_cos_angles_beta[i] = (
                    beta_dist.fit_beta_distribution(
                        geom_fts.edge_pair_cos_angles[m_edge_pair_cos_angles[:, i], j],
                        -1,
                        1,
                    )
                )
                edge_pair_ratios_alpha[i], edge_pair_ratios_beta[i] = (
                    beta_dist.fit_beta_distribution(
                        geom_fts.edge_pair_ratios[m_edge_pair_ratios[:, i], j],
                        0,
                        1,
                    )
                )
            except Exception as e:
                print(
                    f"Failed to fit beta distribution for edge pair {i}",
                    e,
                    torch.sum(m_edge_pair_cos_angles[:, i]),
                )
                break

    for c in tqdm(range(0, n_tets, max_combinations)):
        geom_relation_combinations_batch = data_classes.GeomRelationCombinations(
            e1=geom_relation_combinations.e1[:1],
            e2=geom_relation_combinations.e2[:1],
            t=geom_relation_combinations.t[c : c + max_combinations],
            v=geom_relation_combinations.v[c : c + max_combinations],
        )

        # create masks
        pseudo_geom_fts = geom_features.compute_geom_features_batched(
            vertices=pseudo_kpt_xyz,
            geom_relation_combinations=geom_relation_combinations_batch,
        )
        m_cos_angle_x = ~torch.isnan(pseudo_geom_fts.cos_angle_x)
        m_cos_angle_y = ~torch.isnan(pseudo_geom_fts.cos_angle_y)
        m_cos_angle_z = ~torch.isnan(pseudo_geom_fts.cos_angle_z)
        m_cos_angle[:, c : c + max_combinations] = (
            m_cos_angle_x & m_cos_angle_y & m_cos_angle_z
        )

        # Compute geom_relations of gt_kpts_data
        geom_fts = geom_features.compute_geom_features_batched(
            vertices=gt_kpts_data.xyz,
            geom_relation_combinations=geom_relation_combinations_batch,
        )

        # Fit beta distributions on geom_relations
        for j in range(min(max_combinations, n_tets - c)):
            i = c + j
            if torch.sum(m_cos_angle[:, i]) == 0:
                continue
            try:
                cos_angle_x_alpha[i], cos_angle_x_beta[i] = (
                    beta_dist.fit_beta_distribution(
                        geom_fts.cos_angle_x[m_cos_angle_x[:, j], j], -1, 1
                    )
                )
                cos_angle_y_alpha[i], cos_angle_y_beta[i] = (
                    beta_dist.fit_beta_distribution(
                        geom_fts.cos_angle_y[m_cos_angle_y[:, j], j], -1, 1
                    )
                )
                cos_angle_z_alpha[i], cos_angle_z_beta[i] = (
                    beta_dist.fit_beta_distribution(
                        geom_fts.cos_angle_z[m_cos_angle_z[:, j], j], -1, 1
                    )
                )
            except Exception as e:
                print(
                    f"Failed to fit beta distribution for tet {i}. Note it could be x,y,or z",
                    e,
                    torch.sum(m_cos_angle[:, i]),
                )
                break

    stats = data_classes.GeomRelationStatisticsBeta(
        edge_pair_cos_angles_alpha=edge_pair_cos_angles_alpha,
        edge_pair_cos_angles_beta=edge_pair_cos_angles_beta,
        edge_pair_ratios_alpha=edge_pair_ratios_alpha,
        edge_pair_ratios_beta=edge_pair_ratios_beta,
        cos_angle_x_alpha=cos_angle_x_alpha,
        cos_angle_x_beta=cos_angle_x_beta,
        cos_angle_y_alpha=cos_angle_y_alpha,
        cos_angle_y_beta=cos_angle_y_beta,
        cos_angle_z_alpha=cos_angle_z_alpha,
        cos_angle_z_beta=cos_angle_z_beta,
        m_edge_pair_cos_angles=m_edge_pair_cos_angles,
        m_edge_pair_ratios=m_edge_pair_ratios,
        m_cos_angle=m_cos_angle,
    )
    return stats, geom_relation_combinations
