import torch
from typing import Dict

from semalign3d.core import data_classes


def compute_angles_and_ratios(
    kpts_xyz: torch.Tensor,
):
    """
    Args:
        kpts_xyz: (N, 3)
    Returns:
        angles: (N, N, N) - containing the angles between each triplet of keypoints.
        ratios: (N, N, N) - containing the ratios of distances between each pair of keypoints.
    """
    edges = kpts_xyz[None, :, :] - kpts_xyz[:, None, :]  # (N, N, 3)
    edge_norms = torch.norm(edges, dim=-1)  # (N, N)
    dots = torch.sum(edges[:, :, None, :] * edges[:, None, :, :], dim=-1)  # (N, N, N)
    norm_prods = edge_norms[:, :, None] * edge_norms[:, None, :]  # (N, N, N)
    # NOTE angles[i,i,i] and ratios[i,i,i] is not well defined but that does not matter
    # (angle for i,j,k where |{i,j,k}| <= 2 is also not relevant)
    angles = torch.acos(dots / (norm_prods + 1e-6))  # (N, N, N)
    ratios = edge_norms[:, :, None] / (
        edge_norms[:, :, None] + edge_norms[:, None, :] + 1e-6
    )  # (N, N, N)
    return angles, ratios


def compute_angles_and_ratios_parallel(kpts_xyz: torch.Tensor):
    """
    Computes angles and ratios between keypoints in 3D space across multiple images in parallel.

    This function calculates the pairwise vector differences between keypoints for each image,
    then computes the angles and ratios based on these differences. The computation is vectorized
    and operates on all images in parallel for efficiency.

    Args:
        kpts_xyz (torch.Tensor): A tensor of shape (n_imgs, N, 3) containing the 3D coordinates
                                 of keypoints for each image. `n_imgs` is the number of images,
                                 and `N` is the number of keypoints per image.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing two tensors:
            - Angles: A tensor of shape (n_imgs, N, N, N) representing the angles between each
                      triplet of keypoints for each image.
            - Ratios: A tensor of shape (n_imgs, N, N, N) representing the ratios of distances
                      between each pair of keypoints for each image.
    """
    # Step 1: Compute Edges
    edges = kpts_xyz[:, None, :, :] - kpts_xyz[:, :, None, :]  # (n_imgs, N, N, 3)

    # Step 2: Compute Edge Norms
    edge_norms = torch.norm(edges, dim=-1)  # (n_imgs, N, N)

    # Step 3: Compute Dot Products
    dots = torch.sum(
        edges[:, :, :, None, :] * edges[:, :, None, :, :], dim=-1
    )  # (n_imgs, N, N, N)

    # Step 4: Compute Norm Products
    norm_prods = (
        edge_norms[:, :, :, None] * edge_norms[:, :, None, :]
    )  # (n_imgs, N, N, N)

    # Step 5: Compute Angles
    # TODO do not consider entries where i=j=k
    cos_angles = dots / (norm_prods + 1e-6)  # (n_imgs, N, N, N)
    angles = torch.acos(cos_angles)  # (n_imgs, N, N, N)

    # Step 6: Compute Ratios
    ratios = edge_norms[:, :, :, None] / (
        edge_norms[:, :, :, None] + edge_norms[:, :, None, :] + 1e-6
    )  # (n_imgs, N, N, N)

    # Step 7: Return
    return cos_angles, angles, ratios


def compute_geom_features_weight(
    vertex_weights: torch.Tensor,
    geom_relation_combinations: data_classes.GeomRelationCombinations,
):
    """
    Args:
        vertex_weights: (B, N) tensor containing the weights of the vertices.
        geom_relation_combinations: GeomRelationCombinations
    """
    e1, e2 = geom_relation_combinations.e1, geom_relation_combinations.e2
    t, v = geom_relation_combinations.t, geom_relation_combinations.v

    e1_w = vertex_weights[:, e1]
    e2_w = vertex_weights[:, e2]
    e_w = torch.cat([e1_w, e2_w], dim=2)
    e_w_min = torch.min(e_w, dim=2).values

    t_w = vertex_weights[:, t]
    v_w = vertex_weights[:, v[:, None]]
    tv_w = torch.cat([t_w, v_w], dim=2)
    tv_w_min = torch.min(tv_w, dim=2).values

    return e_w_min, tv_w_min


def compute_geom_features_for_edge_pairs(
    vertices: torch.Tensor,
    geom_relation_combinations: Dict[str, torch.Tensor],
):
    e1, e2 = geom_relation_combinations["e1"], geom_relation_combinations["e2"]
    # Adjust indexing for batch dimension
    edges1 = vertices[:, e1[:, 0]] - vertices[:, e1[:, 1]]
    edges2 = vertices[:, e2[:, 0]] - vertices[:, e2[:, 1]]

    edges1_length = torch.norm(edges1, dim=-1)
    edges2_length = torch.norm(edges2, dim=-1)
    edges1_hat = edges1 / (edges1_length[..., None] + 1e-6)
    edges2_hat = edges2 / (edges2_length[..., None] + 1e-6)
    edge_pair_cos_angles = torch.sum(edges1_hat * edges2_hat, dim=-1)
    edge_pair_ratios = edges1_length / (edges1_length + edges2_length + 1e-6)
    return {
        "edge_pair_cos_angles": edge_pair_cos_angles,
        "edge_pair_ratios": edge_pair_ratios,
    }


def compute_geom_features_for_tets(
    vertices: torch.Tensor,
    geom_relation_combinations: Dict[str, torch.Tensor],
):
    t, v = geom_relation_combinations["t"], geom_relation_combinations["v"]
    # Triangle computations
    triangles = vertices[:, t]  # Adjusted for batch dimension
    edge1 = triangles[:, :, 1] - triangles[:, :, 0]
    edge2 = triangles[:, :, 2] - triangles[:, :, 0]
    origins = triangles[:, :, 0]
    z_axis = torch.cross(edge1, edge2, dim=-1)
    z_axis = z_axis / (torch.norm(z_axis, dim=-1, keepdim=True) + 1e-6)
    x_axis = edge1 / (torch.norm(edge1, dim=-1, keepdim=True) + 1e-6)
    y_axis = torch.cross(z_axis, x_axis, dim=-1)
    x_axis = x_axis / (torch.norm(x_axis, dim=-1, keepdim=True) + 1e-6)
    y_axis = y_axis / (torch.norm(y_axis, dim=-1, keepdim=True) + 1e-6)
    z_axis = z_axis / (torch.norm(z_axis, dim=-1, keepdim=True) + 1e-6)
    # Vertex projections
    verts = vertices[:, v, :]  # Adjusted for batch dimension
    verts = verts - origins
    verts = verts / (torch.norm(verts, dim=-1, keepdim=True) + 1e-6)

    cos_angle_x = torch.sum(verts * x_axis, dim=-1)
    cos_angle_y = torch.sum(verts * y_axis, dim=-1)
    cos_angle_z = torch.sum(verts * z_axis, dim=-1)

    return {
        "cos_angle_x": cos_angle_x,
        "cos_angle_y": cos_angle_y,
        "cos_angle_z": cos_angle_z,
    }


def compute_geom_features_batched_dict(
    vertices: torch.Tensor,
    geom_relation_combinations: Dict[str, torch.Tensor],
):
    """Compute geometric relations for a batch of keypoints.
    Args:
        vertices: (B, N, 3) tensor containing the 3D coordinates of the keypoints.
        geom_relation_combinations: GeomRelationCombinations
    """
    geom_fts_edge_pairs = compute_geom_features_for_edge_pairs(
        vertices, geom_relation_combinations
    )
    geom_fts_tets = compute_geom_features_for_tets(vertices, geom_relation_combinations)
    geom_relation_res = {
        "edge_pair_cos_angles": geom_fts_edge_pairs["edge_pair_cos_angles"],
        "edge_pair_ratios": geom_fts_edge_pairs["edge_pair_ratios"],
        "cos_angle_x": geom_fts_tets["cos_angle_x"],
        "cos_angle_y": geom_fts_tets["cos_angle_y"],
        "cos_angle_z": geom_fts_tets["cos_angle_z"],
    }

    return geom_relation_res


def compute_geom_features_batched(
    vertices: torch.Tensor,
    geom_relation_combinations: data_classes.GeomRelationCombinations,
):
    """Compute geometric relations for a batch of keypoints.
    Args:
        vertices: (B, N, 3) tensor containing the 3D coordinates of the keypoints.
        geom_relation_combinations: GeomRelationCombinations
    """
    res = compute_geom_features_batched_dict(
        vertices,
        {
            "e1": geom_relation_combinations.e1,
            "e2": geom_relation_combinations.e2,
            "t": geom_relation_combinations.t,
            "v": geom_relation_combinations.v,
        },
    )

    geom_relations = data_classes.GeomRelations(
        edge_pair_cos_angles=res["edge_pair_cos_angles"],
        edge_pair_ratios=res["edge_pair_ratios"],
        cos_angle_x=res["cos_angle_x"],
        cos_angle_y=res["cos_angle_y"],
        cos_angle_z=res["cos_angle_z"],
    )

    return geom_relations
