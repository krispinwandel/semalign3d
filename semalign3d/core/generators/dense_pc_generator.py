import os
from typing import Optional, Dict, List
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from semalign3d.core import data_classes
from semalign3d.core.generators.generator import Generator
from semalign3d.core.data import sem3d_data_utils
from semalign3d.utils import image_processing, point_cloud_transforms, cluster_utils


DEFAULT_MERGE_PARAMS = {
    "n_max_pts_per_img": 10000,  # number of points per image to sample
    "max_imgs": None,  # maximum number of images to process (None = all)
}
DEFAULT_CLUSTER_PARAMS = {
    "num_clusters": 1000,  # number of clusters to create
    "n_max_neighbours": 1000,  # number of neighbours to consider for each cluster
}


def load_data(category, paths: data_classes.SemAlign3DPaths, use_vggt=False):
    sem_align_3d_data = sem3d_data_utils.load_data(
        category=category,
        paths=paths,
        load_geom_stats_suffix="",
        sparse_pc_suffix="",
        do_load_dense_pc=False,
        use_vggt=use_vggt,
    )
    return sem_align_3d_data


def downsample(props: Dict[str, torch.Tensor], n_cur, n_max_samples):
    if n_cur <= n_max_samples:
        return props
    indices = torch.randperm(n_cur)[:n_max_samples]
    new_props = {}
    for key, value in props.items():
        new_props[key] = value[indices.to(value.device)]
    return new_props


def transform_img_pc(
    sparse_pc_canonical: torch.Tensor,
    sparse_pc_tets: torch.Tensor,
    kpt_xy: torch.Tensor,
    kpt_labels: torch.Tensor,
    img_xyz: torch.Tensor,
    seg_mask: np.ndarray,
    n_max_pts=10000,
):
    """Transform the image point cloud based on the sparse point cloud and keypoints."""
    device = sparse_pc_canonical.device
    img_shape = seg_mask.shape
    img_coords = image_processing.create_coordinate_tensor(*img_shape, device=device)
    seg_xyz = img_xyz[seg_mask, :]
    seg_xy = img_coords[seg_mask, :]
    n_pts_in_seg_mask = seg_xyz.shape[0]
    indices_sampled = None
    if n_pts_in_seg_mask > n_max_pts:
        indices_sampled = torch.randperm(n_pts_in_seg_mask)[:n_max_pts]
        seg_xyz = seg_xyz[indices_sampled]
        seg_xy = seg_xy[indices_sampled]
    kpt_xyz = torch.full(
        (sparse_pc_canonical.shape[0], 3), torch.inf, dtype=img_xyz.dtype
    ).to(device)
    kpt_xyz[kpt_labels, :] = img_xyz[kpt_xy[:, 1], kpt_xy[:, 0], :]
    noinf = ~kpt_xyz.isinf().any(dim=1)
    try:
        seg_xyz_transformed, _ = point_cloud_transforms.transform_point_cloud(
            kpt_xyz[noinf, :], sparse_pc_canonical[noinf, :], seg_xyz, thresh=1e-1
        )
        # make sure all transformed points are still inside the convex hull of the sparse point cloud
        _, is_in_convex_hull = point_cloud_transforms.points_in_convex_hull(
            seg_xyz_transformed, sparse_pc_tets, thresh=1e-1
        )

    except Exception as e:
        print(f"Error transforming point cloud: {e}")
        # If transformation fails, return empty tensors
        return

    return {
        "xyz_transformed": seg_xyz_transformed[is_in_convex_hull],
        "xyz_orig": seg_xyz[is_in_convex_hull],
        "img_coords": seg_xy[is_in_convex_hull],
    }


def merge_image_point_clouds(
    sparse_pc_canonical: torch.Tensor,
    all_kpt_img_coords: List[torch.Tensor],
    all_img_embds_hat: torch.Tensor,
    all_img_xyz: List[np.ndarray],
    all_seg_masks: List[np.ndarray],
    n_imgs_per_flip: int,
    merge_params=DEFAULT_MERGE_PARAMS,
    max_imgs: Optional[int] = None,
):
    """Merge point clouds from multiple images into a single point cloud."""
    device = sparse_pc_canonical.device
    merge_params = {**DEFAULT_MERGE_PARAMS, **merge_params}

    n_max_pts_per_img = merge_params["n_max_pts_per_img"]
    n_imgs_with_flips = all_img_embds_hat.shape[0]
    assert n_imgs_with_flips == (2 * n_imgs_per_flip)

    merged_pc_xyz = []
    merged_pc_fts_hat = []
    k = sparse_pc_canonical.shape[0]
    img_indices = []
    for i in range(n_imgs_per_flip):
        # make sure we always process flipped image as well even if max_imgs is set
        if (max_imgs is None) or (i < max_imgs):
            img_indices.append(i)
            img_indices.append(i + n_imgs_per_flip)
    _, sparse_pc_tets = point_cloud_transforms.generate_tetrahedrons(
        sparse_pc_canonical
    )
    for i in tqdm(range(len(img_indices))):
        img_idx = img_indices[i]
        # try:
        kpt_xy = all_kpt_img_coords[img_idx][:, :2]
        kpt_labels = all_kpt_img_coords[img_idx][:, 2]
        if len(kpt_labels) < 4:
            # skip images with less than 4 keypoints since we cannot form a tetrahedron
            continue
        img_xyz = torch.from_numpy(all_img_xyz[img_idx]).float().to(device)
        kpt_xyz = torch.full((k, 3), torch.inf, device=device, dtype=img_xyz.dtype)
        kpt_xyz[kpt_labels, :] = img_xyz[kpt_xy[:, 1], kpt_xy[:, 0], :]
        seg_mask = all_seg_masks[img_idx][2]
        transformed_pc = transform_img_pc(
            sparse_pc_canonical=sparse_pc_canonical,
            sparse_pc_tets=sparse_pc_tets,
            kpt_xy=kpt_xy,
            kpt_labels=kpt_labels,
            img_xyz=img_xyz,
            seg_mask=seg_mask,
            n_max_pts=10000,  # high number (we downsample later again)
        )
        transformed_pc = downsample(
            transformed_pc, transformed_pc["xyz_orig"].shape[0], n_max_pts_per_img
        )

        # extract features
        img_embds_hat = all_img_embds_hat[img_idx].to(device)
        embd_size = img_embds_hat.shape[-1]
        img_shape = torch.tensor(img_xyz.shape[:2])
        seg_embd_coords = image_processing.img_coords_to_embd_coords(
            transformed_pc["img_coords"], img_shape, embd_size
        )
        seg_fts_hat = img_embds_hat[:, seg_embd_coords[:, 1], seg_embd_coords[:, 0]]
        seg_fts_hat = seg_fts_hat.T

        # add points and their features to merged point point
        merged_pc_xyz.append(transformed_pc["xyz_transformed"])
        merged_pc_fts_hat.append(seg_fts_hat)
        # except Exception as e:
        #     print(f"Error: {e}")
    merged_pc_xyz = torch.cat(merged_pc_xyz, dim=0)  # (N, 3)
    merged_pc_fts_hat = torch.cat(merged_pc_fts_hat, dim=0)  # (N, F)

    return {
        "xyz": merged_pc_xyz,
        "fts_hat": merged_pc_fts_hat,
    }


def cluster_merged_pc(
    sparse_pc_canonical: torch.Tensor,
    sparse_pc_mask: torch.Tensor,
    points: torch.Tensor,
    point_features: torch.Tensor,
    cluster_params=DEFAULT_CLUSTER_PARAMS,
):
    """Cluster the merged point cloud and compute cluster properties."""
    cluster_params = {**DEFAULT_CLUSTER_PARAMS, **cluster_params}
    num_clusters = cluster_params["num_clusters"]
    n_max_neighbours = cluster_params["n_max_neighbours"]

    # cluster points
    device = points.device
    point_cluster_res = cluster_utils.kmeans_torch(points, num_clusters)
    cluster_labels = point_cluster_res["cluster_labels"].to(device)
    cluster_centers = point_cluster_res["cluster_centers"].to(device)

    # add sparse point cloud as cluster centers
    anchor_xyz = sparse_pc_canonical[sparse_pc_mask]
    n_anchors = anchor_xyz.shape[0]
    cluster_centers = torch.cat(
        [cluster_centers, anchor_xyz], dim=0
    )  # (n_clusters + n_anchors, 3)
    n_combined = num_clusters + n_anchors

    # bary coords
    xyz = cluster_centers
    anchor_tet_combinations, tets = point_cloud_transforms.generate_tetrahedrons(
        sparse_pc_canonical[sparse_pc_mask]
    )
    bary_coords = point_cloud_transforms.barycentric_coordinates(xyz, tets)
    # NOTE anchors are always in convex hull
    # later we only keep clusters that are in the convex hull
    in_convex_hull, is_inside_tet = point_cloud_transforms.bary_coords_in_convex_hull(
        bary_coords, thresh=1e-1
    )
    tet_weights = point_cloud_transforms.tetrahedron_weights(
        xyz, tets, inside_tetrahedrons=is_inside_tet
    )

    # compute cluster properties
    cluster_weights = torch.zeros(n_combined, device=device, dtype=torch.float32)
    cluster_fts_hat = torch.zeros(
        (n_combined, point_features.shape[-1]), device=device, dtype=torch.float32
    )
    cluster_attn_mean = torch.zeros(n_combined, device=device, dtype=torch.float32)
    cluster_attn_var = torch.zeros(n_combined, device=device, dtype=torch.float32)
    for cluster_idx in range(n_combined):
        # cluster weight
        if cluster_idx < num_clusters:
            cluster_weights[cluster_idx] = (
                torch.sum(cluster_labels == cluster_idx) / points.shape[0]
            )
        else:
            # give anchor points a weight of 1.0
            cluster_weights[cluster_idx] = 1.0

        # cluster neighbours
        center = cluster_centers[cluster_idx]
        dists = torch.cdist(center.unsqueeze(0), points, p=2.0)
        nbr_indices = torch.sort(dists[0], descending=False).indices[:n_max_neighbours]
        nbr_fts = point_features[nbr_indices]

        # cluster feature
        cluster_ft_hat = F.normalize(torch.mean(nbr_fts, dim=0), dim=0)

        # attention stats
        attn = torch.sum(cluster_ft_hat[None, :] * nbr_fts, dim=1)
        attn_mean = torch.mean(attn)
        attn_var = torch.mean((attn - attn_mean) ** 2)

        # update
        cluster_fts_hat[cluster_idx] = cluster_ft_hat
        cluster_attn_mean[cluster_idx] = attn_mean
        cluster_attn_var[cluster_idx] = attn_var

    return {
        "centers": cluster_centers[in_convex_hull],
        "weights": cluster_weights[in_convex_hull],
        "fts_hat": cluster_fts_hat[in_convex_hull],
        "attn_mean": cluster_attn_mean[in_convex_hull],
        "attn_var": cluster_attn_var[in_convex_hull],
        # bary coordinates
        # NOTE tet_vertex_indices are with respect to MASKED sparse pc
        "tet_weights": tet_weights[in_convex_hull],
        "bary_coords": bary_coords[in_convex_hull],
        "anchor_tet_combinations": anchor_tet_combinations,
        "anchor_xyz": anchor_xyz,
    }


def construct_dense_pc(
    sparse_pc: torch.Tensor,
    sparse_pc_mask: torch.Tensor,
    processed_data_train: data_classes.SpairProcessedData,
    merge_params=DEFAULT_MERGE_PARAMS,
    cluster_params=DEFAULT_CLUSTER_PARAMS,
):
    merged_pc = merge_image_point_clouds(
        sparse_pc_canonical=sparse_pc,
        all_kpt_img_coords=processed_data_train.kpt_img_coords,
        all_img_embds_hat=processed_data_train.img_embds_hat,
        all_img_xyz=processed_data_train.xyz,
        all_seg_masks=processed_data_train.seg_masks,
        n_imgs_per_flip=len(processed_data_train.img_files),
        merge_params=merge_params,
    )
    # cluster and average features
    clustered_pc = cluster_merged_pc(
        sparse_pc_canonical=sparse_pc,
        sparse_pc_mask=sparse_pc_mask,
        points=merged_pc["xyz"],
        point_features=merged_pc["fts_hat"],
        cluster_params=cluster_params,
    )
    return {
        "merged_pc": merged_pc,
        "clustered_pc": clustered_pc,
    }


class DensePCGenerator(Generator):
    def __init__(
        self,
        paths: data_classes.SemAlign3DPaths,
        max_imgs: Optional[int] = None,
        n_max_pts_per_img=1000,
        use_vggt=False,
    ):
        super().__init__()
        self.paths = paths
        self.max_imgs = max_imgs
        self.n_max_pts_per_img = n_max_pts_per_img
        self.use_vggt = use_vggt

    def save_dense_pc(self, category: str, dense_pc_res: dict):
        out_dir = os.path.join(self.paths.initial_dense_pc_base_dir, category)
        os.makedirs(out_dir, exist_ok=True)
        merged_pc = dense_pc_res["merged_pc"]
        clustered_pc = dense_pc_res["clustered_pc"]
        for k, v in merged_pc.items():
            torch.save(v.cpu(), f"{out_dir}/merged_pc_{k}{self.paths.suffix}.pt")
        for k, v in clustered_pc.items():
            torch.save(v.cpu(), f"{out_dir}/clustered_pc_{k}{self.paths.suffix}.pt")

    def generate(
        self,
        category: str,
        save=True,
    ):
        # load params
        max_imgs = self.max_imgs
        n_max_pts_per_img = self.n_max_pts_per_img

        # load data
        semalign3d_data = load_data(
            category=category,
            paths=self.paths,
            use_vggt=self.use_vggt,
        )
        sparse_pc_canonical = semalign3d_data.kpt_xyz_obj_orig
        sparse_pc_mask = semalign3d_data.used_kpts_mask
        processed_data = semalign3d_data.full_data.processed_data_train

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dense_pc = construct_dense_pc(
            sparse_pc=sparse_pc_canonical.to(device),
            sparse_pc_mask=sparse_pc_mask.to(device),
            processed_data_train=processed_data,
            merge_params={
                "n_max_pts_per_img": n_max_pts_per_img,
                "max_imgs": max_imgs,
            },
            cluster_params=DEFAULT_CLUSTER_PARAMS,
        )

        if save:
            self.save_dense_pc(category, dense_pc)
