from typing import List
import torch
from tqdm import tqdm


def arg_order_vertices_by_closeness(vertices: torch.Tensor, initial_indices: list = []):
    """
    Idea:
        Start with an initial set of vertices, and then iteratively choose the next vertex
        with the smallest distance to the already chosen vertices.
    Args:
        - vertices: (n_vertices, 3) - Tensor containing vertex coordinates
        - initial_indices: List of initially chosen vertex indices
    Returns:
        - ordered_indices: A list of indices representing the order in which the vertices
          should be visited.
    """
    n_vertices = vertices.shape[0]

    if len(initial_indices) < 2:
        # use keypoint indices with smallest distance as initial indices
        dists = torch.cdist(vertices, vertices, p=2)
        dists.fill_diagonal_(torch.inf)
        dists = torch.cdist(vertices, vertices, p=2)
        dists.fill_diagonal_(torch.inf)
        dists_flat = dists.flatten()
        # Handle NaNs (for some reason distance between two inf values
        # is NaN but distance between inf and real is inf)
        # Here we handle case in which two vertices have inf values
        dists_flat[torch.isnan(dists_flat)] = torch.inf
        min_dist_idx = torch.argmin(dists_flat)
        idx1 = min_dist_idx // n_vertices
        idx2 = min_dist_idx % n_vertices
        initial_indices = [int(idx1), int(idx2)]
        # TODO not sure what was the purpose of the code below
        # NOTE we are lucky because 0 and 1 happen to be the closest keypoints
        # min_dists, min_indices = torch.min(dists, dim=1)
        # initial_indices = [int(min_indices[0].item()), int(min_indices[1].item())]

    remaining_indices = list(
        set(range(n_vertices)) - set(initial_indices)
    )  # Remaining indices
    ordered_indices = initial_indices.copy()

    while remaining_indices:
        # Get all chosen vertices so far
        chosen_vertices = vertices[ordered_indices]

        # Find the closest vertex to the set of already chosen vertices
        remaining_vertices = vertices[remaining_indices]

        # Compute the minimum distance from each remaining vertex to the set of chosen vertices
        # Broadcasting over all chosen vertices
        distances = torch.cdist(
            remaining_vertices, chosen_vertices, p=2
        )  # (remaining, chosen)
        distances[torch.isnan(distances)] = torch.inf  # Handle NaNs
        min_distances, _ = torch.min(
            distances, dim=1
        )  # Get the minimum distance for each remaining vertex

        # Choose the vertex with the smallest minimum distance
        next_vertex_idx_in_remaining = int(torch.argmin(min_distances).item())
        next_vertex_idx = remaining_indices[next_vertex_idx_in_remaining]

        # Add the chosen vertex to the ordered list and remove it from remaining indices
        ordered_indices.append(next_vertex_idx)
        remaining_indices.remove(next_vertex_idx)

    return ordered_indices


def find_longest_shortest_edge(
    e1: torch.Tensor,
    e2: torch.Tensor,
    edge_pair_ratios: torch.Tensor,
    n_max_kpts: int = 30,
    find_farthest: bool = True,
):
    """
    Find edge that is larger/smaller than all other edges (based on considered edge pair ratios).
    This

    Args:
        e1: (n_edge_pairs, 2) indices of edge pair verticies of e1 in edge pair (e1, e2)
        e2: (n_edge_pairs, 2) indices of edge pair verticies of e2 in edge pair (e1, e2)
        edge_pair_ratios: (n_edge_pairs,) tensor containing edge ratios of edge_pairs_e1[i] with some other edge e2
        n_max_kpts: int, maximum number of keypoints to consider
        find_farthest: bool, if False, find the closest edge instead of the farthest
    """
    n_e = e1.shape[0]
    ratio_collector = -torch.ones((n_max_kpts, n_max_kpts), dtype=torch.int16)
    for i in tqdm(range(n_e)):
        e1_v1, e1_v2 = e1[i]
        e2_v1, e2_v2 = e2[i]
        # ratio = e1 / (e1 + e2), where e2 can be any edge
        r = edge_pair_ratios[i]
        if not find_farthest:
            r = 1 - r  # invert ratio if we are looking for closest edge
        if r < 0.5:
            ratio_collector[e1_v1, e1_v2] = 0
            ratio_collector[e1_v2, e1_v1] = 0
            if ratio_collector[e2_v1, e2_v2] == -1:
                ratio_collector[e2_v1, e2_v2] = 1
                ratio_collector[e2_v2, e2_v1] = 1
        elif r > 0.5:
            ratio_collector[e2_v1, e2_v2] = 0
            ratio_collector[e2_v2, e2_v1] = 0
            if ratio_collector[e1_v2, e1_v1] == -1:
                ratio_collector[e1_v1, e1_v2] = 1
                ratio_collector[e1_v2, e1_v1] = 1
    # get max indices from ratio_collector
    max_indices = torch.argmax(ratio_collector)
    e1_v1 = max_indices // n_max_kpts
    e1_v2 = max_indices % n_max_kpts
    return int(e1_v1.item()), int(e1_v2.item())


def vertex_is_connected_to_chosen(
    next_kpt_index: int,
    chosen_kpt_indices: List[int],
    e1: torch.Tensor,
    e2: torch.Tensor,
):
    """Filter geometric relations based on chosen keypoints and next keypoint index."""
    # filter geom combinations
    n_e = e1.shape[0]
    e_sum = torch.zeros(n_e, dtype=torch.int16)
    for kpt_index in chosen_kpt_indices:
        e_sum += (
            torch.any(e1 == kpt_index, dim=-1).int()
            + torch.any(e2 == kpt_index, dim=-1).int()
        )
    e_sum_must = (
        torch.any(e1 == next_kpt_index, dim=-1).int()
        + torch.any(e2 == next_kpt_index, dim=-1).int()
    )
    e_filter = (e_sum >= 3) & (e_sum_must >= 1)
    vertex_is_connected = torch.any(e_filter).item()
    return e_filter, vertex_is_connected


def sort_by_distance_to_set(
    initial_vertex_indices: List[int],
    e1: torch.Tensor,
    e2: torch.Tensor,
    edge_pair_ratios: torch.Tensor,
):
    all_vertex_indices = set(e1.flatten().tolist()) | set(e2.flatten().tolist())
    remaining_vertex_indices = list(all_vertex_indices - set(initial_vertex_indices))
    ordered_indices = []
    while remaining_vertex_indices:
        next_vertex_id, next_min_ratio = None, torch.inf
        for r_idx in remaining_vertex_indices:
            e_filter, vertex_is_connected = vertex_is_connected_to_chosen(
                r_idx, ordered_indices + initial_vertex_indices, e1, e2
            )
            if not vertex_is_connected:
                continue
            r_in_v1 = e_filter & torch.any(e1 == r_idx, dim=-1)
            r_in_v2 = e_filter & torch.any(e2 == r_idx, dim=-1)
            # v1 / (v1 + v2) = 1 - (v2 / (v1 + v2))
            r = torch.ones_like(edge_pair_ratios)
            r[r_in_v1] = edge_pair_ratios[r_in_v1]
            r[r_in_v2] = 1 - edge_pair_ratios[r_in_v2]
            min_ratio = torch.min(r)
            if min_ratio < next_min_ratio:
                next_min_ratio = min_ratio
                next_vertex_id = r_idx
        if next_vertex_id is not None:
            ordered_indices.append(next_vertex_id)
            remaining_vertex_indices.remove(next_vertex_id)
        else:
            break
    return ordered_indices
