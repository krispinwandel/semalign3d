import torch
from semalign3d.core import data_classes


def generate_edge_pair_combinations(n: int, full=False):
    """Generate all combinations of distinct pairs of edges for a given number of vertices.
    Args:
        n: number of vertices
    Returns:
        e1: (n_edge_pairs, 2) - containing the vertex indices of the first edge in each pair.
        e2: (n_edge_pairs, 2) - containing the vertex indices of the second edge in each pair.
    """
    edge_combinations = torch.combinations(torch.arange(n), r=2)  # (n_edges, 2)
    edge_pair_combinations = torch.combinations(
        torch.arange(edge_combinations.shape[0]), r=2
    )  # (n_edge_pairs, 2)

    e1 = edge_combinations[edge_pair_combinations[:, 0]]  # (n_edge_pairs, 2)
    e2 = edge_combinations[edge_pair_combinations[:, 1]]  # (n_edge_pairs, 2)

    if full:
        e1_ = torch.cat(
            [e1[:, [0, 1]], e1[:, [1, 0]], e1[:, [0, 1]], e1[:, [1, 0]]], dim=0
        )
        e2_ = torch.cat(
            [e2[:, [0, 1]], e2[:, [0, 1]], e2[:, [1, 0]], e2[:, [1, 0]]], dim=0
        )
        e1 = torch.cat([e1_, e2_], dim=0)
        e2 = torch.cat([e2_, e1_], dim=0)

    return e1, e2


def generate_edge_pair_combinations_partly(n: int):
    """Assumes that vertex at index 0 is fixed. And other vertices go from 1 to n-1."""
    if n == 3:
        e1 = torch.tensor([[0, 1], [0, 2]])
        e2 = torch.tensor([[1, 2], [1, 2]])
    elif n > 3:
        triangles = torch.combinations(1 + torch.arange(n - 1), r=3)  # (n_triangles, 3)
        v0 = torch.zeros(triangles.shape[0], dtype=torch.int64)

        # v0-i, i-j where 0 <= i,j < 3 & i != j
        e1_0 = torch.stack([v0, triangles[:, 0]], dim=-1)
        e1_1 = torch.stack([v0, triangles[:, 0]], dim=-1)
        e1_2 = torch.stack([v0, triangles[:, 1]], dim=-1)
        e1_3 = torch.stack([v0, triangles[:, 1]], dim=-1)
        e1_4 = torch.stack([v0, triangles[:, 2]], dim=-1)
        e1_5 = torch.stack([v0, triangles[:, 2]], dim=-1)
        # v0-j, k-l where 0 <= j,k,l < 3 & j != k != l
        e1_6 = torch.stack([v0, triangles[:, 0]], dim=-1)
        e1_7 = torch.stack([v0, triangles[:, 1]], dim=-1)
        e1_8 = torch.stack([v0, triangles[:, 2]], dim=-1)

        e2_0 = torch.stack([triangles[:, 0], triangles[:, 1]], dim=-1)
        e2_1 = torch.stack([triangles[:, 0], triangles[:, 2]], dim=-1)
        e2_2 = torch.stack([triangles[:, 1], triangles[:, 0]], dim=-1)
        e2_3 = torch.stack([triangles[:, 1], triangles[:, 2]], dim=-1)
        e2_4 = torch.stack([triangles[:, 2], triangles[:, 0]], dim=-1)
        e2_5 = torch.stack([triangles[:, 2], triangles[:, 1]], dim=-1)

        e2_6 = torch.stack([triangles[:, 1], triangles[:, 2]], dim=-1)
        e2_7 = torch.stack([triangles[:, 2], triangles[:, 0]], dim=-1)
        e2_8 = torch.stack([triangles[:, 0], triangles[:, 1]], dim=-1)

        e1 = torch.cat([e1_0, e1_1, e1_2, e1_3, e1_4, e1_5, e1_6, e1_7, e1_8], dim=0)
        e2 = torch.cat([e2_0, e2_1, e2_2, e2_3, e2_4, e2_5, e2_6, e2_7, e2_8], dim=0)
    else:
        raise ValueError("Invalid number of vertices")

    return e1, e2


def cyclic_shift_vector(vec, num_shifts):
    n = vec.size(0)
    shifts = torch.arange(num_shifts).unsqueeze(1)  # Shape: [num_shifts, 1]
    indices = (torch.arange(n) + shifts) % n  # Broadcasting to generate all indices
    shifted_tensors = vec[indices]  # Indexing vec with generated indices
    return shifted_tensors


def generate_trianlge_vertex_combinations(n: int, full=False):
    """Generate all combinations of distinct triangle + vertex pairs for a given number of vertices.
    Args:
        n: number of vertices
    Returns:
        triangles: (n_triangles, 3) tensor containing the vertex indices of each triangle.
        vertices: (n_triangles, 1) tensor containing the vertex index that is not part of the triangle.
    """
    tet_combinations = torch.combinations(torch.arange(n), r=4)  # (n_tets, 4)
    tet_triangle_combinations = cyclic_shift_vector(torch.arange(4), 4)  # (4, 4)
    triangles_vertex = tet_combinations[:, tet_triangle_combinations]  # (n_tets, 4, 4)
    triangles_vertex = triangles_vertex.reshape(-1, 4)
    triangles = triangles_vertex[:, :3]
    vertices = triangles_vertex[:, 3]

    if full:
        triangles = torch.cat(
            [
                triangles[:, [0, 1, 2]],
                triangles[:, [0, 2, 1]],
                triangles[:, [1, 0, 2]],
                triangles[:, [2, 0, 1]],
                triangles[:, [1, 2, 0]],
                triangles[:, [2, 1, 0]],
            ],
            dim=0,
        )
        vertices = vertices.repeat(6)

    return triangles, vertices


def generate_trianlge_vertex_combinations_partly(n: int):
    """Assumes that vertex at index 0 is fixed. And other vertices go from 1 to n-1."""
    triangles = torch.combinations(1 + torch.arange(n - 1), r=3)  # (n_tri, 3)
    vertices = torch.zeros(triangles.shape[0], dtype=torch.int64)
    return triangles, vertices


def generate_geometric_relation_combinations(n: int, full=False):
    e1, e2 = generate_edge_pair_combinations(n, full=full)
    t, v = generate_trianlge_vertex_combinations(n, full=full)
    return data_classes.GeomRelationCombinations(e1=e1, e2=e2, t=t, v=v)


def generate_geometric_relation_combinations_partly(n: int):
    e1, e2 = generate_edge_pair_combinations_partly(n)
    t, v = generate_trianlge_vertex_combinations_partly(n)
    return data_classes.GeomRelationCombinations(e1=e1, e2=e2, t=t, v=v)
