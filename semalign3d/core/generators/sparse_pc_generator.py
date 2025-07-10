import os
from typing import List, Optional, Dict
from tqdm import tqdm
from dataclasses import asdict
import time

import torch
import torch.nn.functional as F

from semalign3d.core import data_classes
from semalign3d.core.geometry import geom_optimizer, geom_filter
from semalign3d.core.losses import geom_loss
from semalign3d.core.data import sem3d_data_utils
from semalign3d.utils.geometry import topology, shapes
from semalign3d.utils import opt_utils
from semalign3d.utils.stats import beta_dist
from semalign3d.core.generators.generator import Generator


DEFAULT_SPARSE_PC_CONSTRUCTION_PARAMS = {
    "cube_size": 1.0,
    "cube_n_points_per_dim": 100,
    "batch_size": 1000,
    "n_max_comb": 100,
}


class SparsePCGenerator(Generator):
    """Sparse Point Cloud Generator for keypoint generation"""

    def __init__(
        self,
        paths: data_classes.SemAlign3DPaths,
        sparse_pc_params: dict = DEFAULT_SPARSE_PC_CONSTRUCTION_PARAMS,
        use_vggt: bool = False,
        geom_stats_suffix: str = "",
        verbose=False,
    ):
        self.paths = paths
        self.sparse_pc_params = DEFAULT_SPARSE_PC_CONSTRUCTION_PARAMS
        self.sparse_pc_params.update(sparse_pc_params)
        self.verbose = verbose
        self.use_vggt = use_vggt
        self.geom_stats_suffix = geom_stats_suffix
        self.geom_optimizer = geom_optimizer.GeomOptimizer(
            opt_params={"n_iter": 20000, "lr": 1e-3}
        )

    def _save(
        self,
        sparse_pc: Dict[str, torch.Tensor],
        category: str,
    ):
        """Save the generated sparse point cloud"""
        out_dir = f"{self.paths.initial_pc_base_dir}/{category}"
        os.makedirs(out_dir, exist_ok=True)
        prefix = "sparse_pc"
        suffix = self.geom_stats_suffix + self.paths.suffix
        for k, v in sparse_pc.items():
            torch.save(v.detach().cpu(), f"{out_dir}/{prefix}_{k}{suffix}.pt")

    def generate(
        self,
        category: str,
        save=True,
    ):
        semalgin3d_data = load_data(
            category=category,
            paths=self.paths,
            use_vggt=self.use_vggt,
            geom_stats_suffix=self.geom_stats_suffix,
        )
        n_max_kpts = semalgin3d_data.n_max_kpts

        self._log("Step 0: construct sparse pc features", category)
        sparse_pc_fts = construct_sparse_pc_features(
            preprocessed_data_train=semalgin3d_data.full_data.processed_data_train,
            n_max_kpts=n_max_kpts,
        )

        self._log("Step 1: constructing", category)

        kpt_xyz_obj = construct_sparse_pc(
            unused_kpts=semalgin3d_data.unused_kpts,
            geom_relation_combinations=semalgin3d_data.geom_relation_combinations_partial,
            geom_stats=semalgin3d_data.geom_stats_partial,
            n_max_kpts=n_max_kpts,
            **self.sparse_pc_params,
        )

        # NOTE iterative sampling leads to compounding errors
        # => so we optimize the point cloud as a whole in post
        self._log("Step 2: optimising", category)
        kpt_xyz_obj = self.geom_optimizer.optimize(
            kpt_xyz_obj,
            geom_stats=semalgin3d_data.geom_stats_partial,
            geom_relation_combinations=semalgin3d_data.geom_relation_combinations_partial,
        )

        if save:
            self._save({"xyz": kpt_xyz_obj, **sparse_pc_fts}, category)

        return kpt_xyz_obj


def load_data(
    category: str,
    paths: data_classes.SemAlign3DPaths,
    use_vggt: bool = False,
    geom_stats_suffix: str = "",
):
    sem_align_3d_data = sem3d_data_utils.load_data(
        category=category,
        paths=paths,
        load_geom_stats_suffix=geom_stats_suffix,
        do_load_sparse_pc=False,
        do_load_dense_pc=False,
        use_vggt=use_vggt,
    )
    return sem_align_3d_data


def get_free_kpt_indices(chosen_kpt_indices: List[int], n_max_kpts: int):
    """Get free keypoint indices"""
    all_kpt_indices = list(range(n_max_kpts))
    free_kpt_indices = list(set(all_kpt_indices) - set(chosen_kpt_indices))
    return free_kpt_indices


def find_pos_for_next_kpt(
    next_kpt_index: int,
    chosen_kpt_indices: List[int],
    kpt_xyz_obj: torch.Tensor,
    geom_relation_combinations: data_classes.GeomRelationCombinations,
    geom_stats_beta: data_classes.GeomRelationStatisticsBetaSimple,
    cube_size: float = 2.0,
    cube_n_points_per_dim=200,
    batch_size=1000,
    n_max_comb=400,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
):
    # filter geom combinations
    filtered_geom_stats_beta, filtered_geom_relation_combinations = (
        geom_filter.filter_geom_relations_chosen_connected_with_next(
            next_kpt_index,
            chosen_kpt_indices,
            geom_stats_beta,
            geom_relation_combinations,
            n_max_comb=n_max_comb,
        )
    )
    constants = {
        "geom_stats": asdict(filtered_geom_stats_beta),
        "geom_relation_combinations": asdict(filtered_geom_relation_combinations),
        "use_tets": len(chosen_kpt_indices) >= 3,
    }
    loss_fun = opt_utils.build_loss(
        loss_fun=geom_loss.calculate_geom_loss,
        constants=constants,
        device=device,
        compile=False,
    )

    # create kpt_xyz_obj_batched
    # next_kpt_index can be anywhere on a cube grid centered at around the object
    kpt_xyz_obj_not_inf = kpt_xyz_obj[~torch.isinf(kpt_xyz_obj).any(dim=-1), :]
    # TODO perhaps use weighted mean of the last few keypoints
    center = kpt_xyz_obj_not_inf.mean(dim=0)
    # TODO instead of a cube grid, we should have high resolution nearby and low resolution farther away

    next_kpt_candidate_positions = shapes.get_cube_grid(
        center, cube_size, cube_n_points_per_dim
    )
    energies = torch.zeros(cube_n_points_per_dim**3, dtype=torch.float32)

    # handle case where keypoint is not present in any of the combinations => return center
    if filtered_geom_relation_combinations.e1.shape[0] == 0:
        return center, 0.0, energies

    # calculate energies for each candidate position
    energies = energies.to(device)
    for i in tqdm(range(0, next_kpt_candidate_positions.shape[0], batch_size)):
        # build kpt_xyz_obj_batched
        next_kpt_candidate_positions_batched = next_kpt_candidate_positions[
            i : i + batch_size
        ]
        real_batch_size = next_kpt_candidate_positions_batched.shape[0]
        kpt_xyz_obj_batched = kpt_xyz_obj[None, :, :]
        kpt_xyz_obj_batched = kpt_xyz_obj_batched.repeat(real_batch_size, 1, 1)
        kpt_xyz_obj_batched[:, next_kpt_index, :] = next_kpt_candidate_positions_batched

        # calculate energies
        energies[i : i + real_batch_size] = loss_fun(
            {"kpt_xyz_obj": kpt_xyz_obj_batched.to(device)}, None, None
        )

    # find position with minimum energy
    energies = energies.to("cpu")
    min_energy, min_energy_index = torch.min(energies, dim=0)
    min_energy_position = next_kpt_candidate_positions[min_energy_index]
    return min_energy_position, min_energy, energies


def construct_sparse_pc(
    n_max_kpts: int,
    unused_kpts: torch.Tensor,
    geom_relation_combinations: data_classes.GeomRelationCombinations,
    geom_stats: data_classes.GeomRelationStatisticsBetaSimple,
    # sparse pc construction params
    cube_size: float = 2.0,
    cube_n_points_per_dim: int = 100,
    batch_size: int = 1000,
    n_max_comb: int = 100,
):
    edge_pair_ratios = beta_dist.compute_beta_mode_torch(
        geom_stats.edge_pair_ratios_alpha, geom_stats.edge_pair_ratios_beta
    )
    v1, v2 = topology.find_longest_shortest_edge(
        geom_relation_combinations.e1,
        geom_relation_combinations.e2,
        edge_pair_ratios,
        find_farthest=False,
    )
    print("v1, v2:", v1, v2)
    t1 = time.time()
    next_kpt_indices = topology.sort_by_distance_to_set(
        initial_vertex_indices=[v1, v2],
        e1=geom_relation_combinations.e1,
        e2=geom_relation_combinations.e2,
        edge_pair_ratios=edge_pair_ratios,
    )
    t2 = time.time()
    print(
        f"Sorting by distance took {t2 - t1:.2f} seconds. Number of edges: {geom_relation_combinations.e1.shape[0]}"
    )
    v1_pos = torch.tensor([0.01, 0.01, 0.01])
    v2_pos = -v1_pos
    new_kpt_xyz_obj = torch.full((n_max_kpts, 3), torch.inf, dtype=torch.float32)
    new_kpt_xyz_obj[v1] = v1_pos
    new_kpt_xyz_obj[v2] = v2_pos
    start_kpt_indices = [v1, v2]

    # remove unused keypoints from next_kpt_indices
    unused_kpts = unused_kpts.tolist()
    next_kpt_indices = [
        kpt_index for kpt_index in next_kpt_indices if kpt_index not in unused_kpts
    ]
    print("next_kpt_indices:", next_kpt_indices)

    for kpt_index in next_kpt_indices:
        print("kpt_index", kpt_index)
        min_energy_position, min_energy, energies = find_pos_for_next_kpt(
            next_kpt_index=kpt_index,
            chosen_kpt_indices=start_kpt_indices,
            kpt_xyz_obj=new_kpt_xyz_obj,
            geom_relation_combinations=geom_relation_combinations,
            geom_stats_beta=geom_stats,
            cube_size=cube_size,
            cube_n_points_per_dim=cube_n_points_per_dim,
            batch_size=batch_size,
            n_max_comb=n_max_comb,
        )
        new_kpt_xyz_obj[kpt_index] = min_energy_position
        start_kpt_indices.append(kpt_index)
    return new_kpt_xyz_obj


def construct_sparse_pc_features(
    preprocessed_data_train: data_classes.SpairProcessedData,
    n_max_kpts=30,
):
    """Compute sparse pc features"""
    # average keypoint embeddings
    kpt_features_avg_train = []
    kpt_features_attn_avg_train = []
    kpt_features_attn_sd_train = []
    C = preprocessed_data_train.img_embds.shape[1]
    for kpt_idx in range(n_max_kpts):
        kpt_features_hat = preprocessed_data_train.kpt_idx_to_kpt_embds_hat[kpt_idx]
        # kpt_features is only None if keypoint label does not exist for object category
        if kpt_features_hat is None:
            # kpt does not exist
            kpt_features_avg_train.append(torch.zeros((1, C), dtype=torch.float32))
            kpt_features_attn_avg_train.append(torch.tensor([[0.5]]))
            kpt_features_attn_sd_train.append(torch.tensor([[0.01]]))
            continue

        kpt_features_avg = torch.mean(kpt_features_hat, dim=0, keepdim=True)  # (1, C)
        kpt_features_avg = F.normalize(
            kpt_features_avg, dim=1
        )  # normalize the average feature
        # compute dot product between kpt_features and kpt_features_avg
        kpt_features_attn = torch.bmm(
            kpt_features_hat.unsqueeze(0), kpt_features_avg.unsqueeze(2)
        ).squeeze(
            0
        )  # (N,1)
        kpt_features_attn_avg = torch.mean(
            kpt_features_attn, dim=0, keepdim=True
        )  # (1,1)
        kpt_features_attn_sd = torch.std(
            kpt_features_attn, dim=0, keepdim=True
        )  # (1,1)
        kpt_features_avg_train.append(kpt_features_avg)
        kpt_features_attn_avg_train.append(kpt_features_attn_avg)
        kpt_features_attn_sd_train.append(kpt_features_attn_sd)

    kpt_features_avg_train = torch.cat(kpt_features_avg_train, dim=0)
    kpt_features_attn_avg_train = torch.cat(kpt_features_attn_avg_train, dim=0)
    kpt_features_attn_sd_train = torch.cat(kpt_features_attn_sd_train, dim=0)

    return {
        "fts_hat": kpt_features_avg_train,
        "attn_mean": kpt_features_attn_avg_train[:, 0],
        "attn_sd": kpt_features_attn_sd_train[:, 0],
    }
