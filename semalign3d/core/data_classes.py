"""SemAlign3D data classes"""

import torch
from dataclasses import dataclass
from typing import List
import numpy as np


@dataclass
class ImgAnnoNormalized:
    bbox: List[int]
    kp_ids: List[int]
    kp_xy: List[List[int]]
    filename: str
    img_width: int
    img_height: int
    category: str
    supercategory: str
    rel_fp: str


@dataclass
class SrcTrgImgAnno:
    src: ImgAnnoNormalized
    trg: ImgAnnoNormalized


@dataclass
class SemAlign3DPaths:
    dataset_name: str
    dataset_name_short: str
    semalign3d_data_dir: str
    spair_data_dir: str
    img_file_splits_dir: str
    geom_stats_dir: str
    img_focal_lengths_opt_dir: str
    initial_pc_base_dir: str
    initial_dense_pc_base_dir: str
    depths_dir: str
    kpt_optimised_depth_dir: str
    seg_masks_dir: str
    seg_masks_inter_dir: str
    sam_vit_h_ckpt_path: str
    embds_folder: str
    aggre_net_ckpt_path: str
    dense_pc_suffix: str
    img_fitted_models_dir: str
    scores_dir: str
    vggt_dir: str
    suffix: str
    depth_generator: str

    def __init__(
        self,
        dataset_name: str,
        semalign3d_data_dir: str,
        spair_data_dir: str,
        embds_folder: str,
        aggre_net_ckpt_path: str,
        sam_vit_h_ckpt_path: str = None,
        dense_pc_suffix: str = "",
        suffix: str = "",
        depth_generator: str = "vggt",
    ):
        dataname_to_short = {
            "spair-71k": "spair",
            "ap-10k": "ap10k",
            "pf-pascal": "pascal",
        }
        assert (
            dataset_name in dataname_to_short
        ), f"Invalid dataset name: {dataset_name}, allowed: {dataname_to_short.keys()}"
        self.dataset_name = dataset_name
        self.dataset_name_short = dataname_to_short[dataset_name]
        self.semalign3d_data_dir = semalign3d_data_dir
        self.spair_data_dir = spair_data_dir
        self.embds_folder = embds_folder
        self.sam_vit_h_ckpt_path = sam_vit_h_ckpt_path
        self.aggre_net_ckpt_path = aggre_net_ckpt_path
        self.depths_dir = f"{semalign3d_data_dir}/DepthAnythingV2"
        self.kpt_optimised_depth_dir = f"{semalign3d_data_dir}/KeypointOptimisedDepth"
        self.seg_masks_dir = f"{semalign3d_data_dir}/SegAnyMasks"
        self.seg_masks_inter_dir = f"{semalign3d_data_dir}/SegAnyMasksInter"
        self.img_file_splits_dir = f"{semalign3d_data_dir}/ImgFileSplits"
        self.geom_stats_dir = f"{semalign3d_data_dir}/GeomBetaStatistics"
        self.img_focal_lengths_opt_dir = f"{semalign3d_data_dir}/ImgFocalLengthsOpt"
        self.initial_pc_base_dir = f"{semalign3d_data_dir}/InitialPointClouds"
        self.initial_dense_pc_base_dir = (
            f"{semalign3d_data_dir}/DenseInitialPointClouds"
        )
        self.img_fitted_models_dir = f"{semalign3d_data_dir}/ImgFittedModels"
        self.scores_dir = f"{semalign3d_data_dir}/Scores"
        self.dense_pc_suffix = dense_pc_suffix
        self.vggt_dir = f"{semalign3d_data_dir}/VGGT"
        self.suffix = suffix
        self.depth_generator = depth_generator


@dataclass
class SpairDataConfig:
    spair_data_folder: str
    embds_folder: str
    img_files_np_path_train: str
    img_files_np_path_eval: str
    depth_folder: str
    masks_folder: str
    vggt_folder: str
    focal_lengths_inv_file_train: str
    focal_lengths_inv_file_eval: str
    img_size: int
    embd_size: int
    pad: bool
    flips_train: List[bool]
    flips_eval: List[bool]
    aggre_net_fp: str
    embd_type: str  # geo or geo_agg
    depth_type: str  # vggt or depth_any


@dataclass
class EvalConfig:
    category: str


@dataclass
class SpairProcessedData:
    img_files: List[str]
    img_embds: torch.Tensor
    img_embds_hat: torch.Tensor
    kpt_idx_to_kpt_embds_hat: dict
    kpt_embd_coords: List[torch.Tensor]
    kpt_img_coords: List[torch.Tensor]
    depths: List[np.ndarray]
    xyz: List[np.ndarray]
    seg_masks: List[np.ndarray]
    seg_auto_masks: List[np.ndarray]
    n_max_kpts: int
    unused_kpts: torch.Tensor
    # NOTE focal_lengths_inv is not populated when using VGGT
    focal_lengths_inv: np.ndarray
    # additional props for VGGT
    normalized_camera_intrinsics: np.ndarray
    depths_conf: List[np.ndarray]


# NOTE will be removed in a future release
@dataclass
class SparsePCFeatures:
    kpt_features_avg_hat: torch.Tensor
    kpt_features_attn_avg: torch.Tensor
    kpt_features_attn_sd: torch.Tensor


# NOTE will be removed in a future release
@dataclass
class SpairFullData:
    processed_data_train: SpairProcessedData
    processed_data_eval: SpairProcessedData
    keypoint_features: SparsePCFeatures


@dataclass
class GtKptsData:
    xyz: torch.Tensor
    vertex_mask: torch.Tensor


@dataclass
class GeomRelationCombinations:
    e1: torch.Tensor
    e2: torch.Tensor
    t: torch.Tensor
    v: torch.Tensor

    def __init__(self, e1, e2, t, v):
        super(GeomRelationCombinations, self).__init__()
        self.e1 = e1
        self.e2 = e2
        self.t = t
        self.v = v

    def to(self, device: str):
        return GeomRelationCombinations(
            e1=self.e1.to(device),
            e2=self.e2.to(device),
            t=self.t.to(device),
            v=self.v.to(device),
        )


@dataclass
class GeomRelations:
    edge_pair_cos_angles: torch.Tensor
    edge_pair_ratios: torch.Tensor
    cos_angle_x: torch.Tensor
    cos_angle_y: torch.Tensor
    cos_angle_z: torch.Tensor

    def __init__(
        self,
        edge_pair_cos_angles,
        edge_pair_ratios,
        cos_angle_x,
        cos_angle_y,
        cos_angle_z,
    ):
        super(GeomRelations, self).__init__()
        self.edge_pair_cos_angles = edge_pair_cos_angles
        self.edge_pair_ratios = edge_pair_ratios
        self.cos_angle_x = cos_angle_x
        self.cos_angle_y = cos_angle_y
        self.cos_angle_z = cos_angle_z

    def to(self, device: str):
        return GeomRelations(
            edge_pair_cos_angles=self.edge_pair_cos_angles.to(device),
            edge_pair_ratios=self.edge_pair_ratios.to(device),
            cos_angle_x=self.cos_angle_x.to(device),
            cos_angle_y=self.cos_angle_y.to(device),
            cos_angle_z=self.cos_angle_z.to(device),
        )


@dataclass
class GeomRelationStatisticsBeta:
    edge_pair_cos_angles_alpha: torch.Tensor
    edge_pair_cos_angles_beta: torch.Tensor
    edge_pair_ratios_alpha: torch.Tensor
    edge_pair_ratios_beta: torch.Tensor
    cos_angle_x_alpha: torch.Tensor
    cos_angle_x_beta: torch.Tensor
    cos_angle_y_alpha: torch.Tensor
    cos_angle_y_beta: torch.Tensor
    cos_angle_z_alpha: torch.Tensor
    cos_angle_z_beta: torch.Tensor
    # masks
    m_edge_pair_cos_angles: torch.Tensor
    m_edge_pair_ratios: torch.Tensor
    m_cos_angle: torch.Tensor

    def __init__(
        self,
        edge_pair_cos_angles_alpha,
        edge_pair_cos_angles_beta,
        edge_pair_ratios_alpha,
        edge_pair_ratios_beta,
        cos_angle_x_alpha,
        cos_angle_x_beta,
        cos_angle_y_alpha,
        cos_angle_y_beta,
        cos_angle_z_alpha,
        cos_angle_z_beta,
        m_edge_pair_cos_angles,
        m_edge_pair_ratios,
        m_cos_angle,
    ):
        super(GeomRelationStatisticsBeta, self).__init__()
        self.edge_pair_cos_angles_alpha = edge_pair_cos_angles_alpha
        self.edge_pair_cos_angles_beta = edge_pair_cos_angles_beta
        self.edge_pair_ratios_alpha = edge_pair_ratios_alpha
        self.edge_pair_ratios_beta = edge_pair_ratios_beta
        self.cos_angle_x_alpha = cos_angle_x_alpha
        self.cos_angle_x_beta = cos_angle_x_beta
        self.cos_angle_y_alpha = cos_angle_y_alpha
        self.cos_angle_y_beta = cos_angle_y_beta
        self.cos_angle_z_alpha = cos_angle_z_alpha
        self.cos_angle_z_beta = cos_angle_z_beta
        self.m_edge_pair_cos_angles = m_edge_pair_cos_angles
        self.m_edge_pair_ratios = m_edge_pair_ratios
        self.m_cos_angle = m_cos_angle

    def to(self, device: str):
        return GeomRelationStatisticsBeta(
            edge_pair_cos_angles_alpha=self.edge_pair_cos_angles_alpha.to(device),
            edge_pair_cos_angles_beta=self.edge_pair_cos_angles_beta.to(device),
            edge_pair_ratios_alpha=self.edge_pair_ratios_alpha.to(device),
            edge_pair_ratios_beta=self.edge_pair_ratios_beta.to(device),
            cos_angle_x_alpha=self.cos_angle_x_alpha.to(device),
            cos_angle_x_beta=self.cos_angle_x_beta.to(device),
            cos_angle_y_alpha=self.cos_angle_y_alpha.to(device),
            cos_angle_y_beta=self.cos_angle_y_beta.to(device),
            cos_angle_z_alpha=self.cos_angle_z_alpha.to(device),
            cos_angle_z_beta=self.cos_angle_z_beta.to(device),
            m_edge_pair_cos_angles=self.m_edge_pair_cos_angles.to(device),
            m_edge_pair_ratios=self.m_edge_pair_ratios.to(device),
            m_cos_angle=self.m_cos_angle.to(device),
        )


@dataclass
class GeomRelationStatisticsBetaSimple:
    edge_pair_cos_angles_alpha: torch.Tensor
    edge_pair_cos_angles_beta: torch.Tensor
    edge_pair_ratios_alpha: torch.Tensor
    edge_pair_ratios_beta: torch.Tensor
    cos_angle_x_alpha: torch.Tensor
    cos_angle_x_beta: torch.Tensor
    cos_angle_y_alpha: torch.Tensor
    cos_angle_y_beta: torch.Tensor
    cos_angle_z_alpha: torch.Tensor
    cos_angle_z_beta: torch.Tensor

    def __init__(
        self,
        edge_pair_cos_angles_alpha,
        edge_pair_cos_angles_beta,
        edge_pair_ratios_alpha,
        edge_pair_ratios_beta,
        cos_angle_x_alpha,
        cos_angle_x_beta,
        cos_angle_y_alpha,
        cos_angle_y_beta,
        cos_angle_z_alpha,
        cos_angle_z_beta,
    ):
        super(GeomRelationStatisticsBetaSimple, self).__init__()
        self.edge_pair_cos_angles_alpha = edge_pair_cos_angles_alpha
        self.edge_pair_cos_angles_beta = edge_pair_cos_angles_beta
        self.edge_pair_ratios_alpha = edge_pair_ratios_alpha
        self.edge_pair_ratios_beta = edge_pair_ratios_beta
        self.cos_angle_x_alpha = cos_angle_x_alpha
        self.cos_angle_x_beta = cos_angle_x_beta
        self.cos_angle_y_alpha = cos_angle_y_alpha
        self.cos_angle_y_beta = cos_angle_y_beta
        self.cos_angle_z_alpha = cos_angle_z_alpha
        self.cos_angle_z_beta = cos_angle_z_beta

    def to(self, device: str):
        return GeomRelationStatisticsBetaSimple(
            edge_pair_cos_angles_alpha=self.edge_pair_cos_angles_alpha.to(device),
            edge_pair_cos_angles_beta=self.edge_pair_cos_angles_beta.to(device),
            edge_pair_ratios_alpha=self.edge_pair_ratios_alpha.to(device),
            edge_pair_ratios_beta=self.edge_pair_ratios_beta.to(device),
            cos_angle_x_alpha=self.cos_angle_x_alpha.to(device),
            cos_angle_x_beta=self.cos_angle_x_beta.to(device),
            cos_angle_y_alpha=self.cos_angle_y_alpha.to(device),
            cos_angle_y_beta=self.cos_angle_y_beta.to(device),
            cos_angle_z_alpha=self.cos_angle_z_alpha.to(device),
            cos_angle_z_beta=self.cos_angle_z_beta.to(device),
        )


@dataclass
class SparsePC:
    xyz: torch.Tensor
    fts_hat: torch.Tensor
    attn_mean: torch.Tensor
    attn_sd: torch.Tensor


@dataclass
class DensePC:
    xyz: torch.Tensor
    fts_hat: torch.Tensor
    attn_mean: torch.Tensor
    attn_var: torch.Tensor

    bary_coords: torch.Tensor
    tet_weights: torch.Tensor

    anchor_xyz: torch.Tensor
    anchor_tet_combinations: torch.Tensor
    closest_to_anchor: torch.Tensor

    color: torch.Tensor
    cmax: torch.Tensor
    cmin: torch.Tensor
    pc_fts_pca: torch.Tensor
    pc_fts_mean: torch.Tensor
    pc_fts_std: torch.Tensor
    pc_fts_sorted_eigenvalues: torch.Tensor
    pc_fts_sorted_eigenvectors: torch.Tensor


@dataclass
class SemAlign3DData:
    full_data: SpairFullData
    geom_stats_partial: GeomRelations
    geom_relation_combinations_partial: torch.Tensor
    n_max_kpts: int
    unused_kpts: torch.Tensor
    dense_pc: DensePC
    kpt_xyz_obj_orig: torch.Tensor
    used_kpts_mask: torch.Tensor
