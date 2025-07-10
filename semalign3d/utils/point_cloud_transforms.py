import torch


# Step 1: Generate all possible tetrahedrons from the skeleton points using torch.combinations
def generate_tetrahedrons(skeleton_points):
    """
    Given a set of skeleton points, generate all tetrahedrons (combinations of 4 points).
    """
    indices = torch.combinations(torch.arange(skeleton_points.shape[0]), r=4).to(
        skeleton_points.device
    )  # Shape: [N_combinations, 4]
    tetrahedrons = skeleton_points[
        indices
    ]  # Index into skeleton points to get the actual tetrahedron vertices
    return indices, tetrahedrons  # Shape: [N_combinations, 4, 3]


# Step 2: Compute generalized barycentric coordinates for a point cloud and multiple tetrahedrons
def barycentric_coordinates(point_cloud, tetrahedrons):
    """
    Compute the generalized barycentric coordinates for all points in the point cloud
    with respect to multiple tetrahedrons.

    point_cloud: Tensor of shape [N_points, 3]
    tetrahedrons: Tensor of shape [N_tetrahedrons, 4, 3]

    Returns:
    - Barycentric coordinates: Tensor of shape [N_points, N_tetrahedrons, 4]
    """
    v0 = tetrahedrons[:, 0, :]  # Shape: [N_tetrahedrons, 3]
    v1 = tetrahedrons[:, 1, :]  # Shape: [N_tetrahedrons, 3]
    v2 = tetrahedrons[:, 2, :]  # Shape: [N_tetrahedrons, 3]
    v3 = tetrahedrons[:, 3, :]  # Shape: [N_tetrahedrons, 3]

    T = torch.stack(
        [v1 - v0, v2 - v0, v3 - v0], dim=-1
    )  # Shape: [N_tetrahedrons, 3, 3]

    # Solve the system T * barycentric_coords = point - v0 for each tetrahedron and point
    relative_points = point_cloud.unsqueeze(1) - v0.unsqueeze(
        0
    )  # Shape: [N_points, N_tetrahedrons, 3]

    # We need to solve this equation for each tetrahedron individually
    # Use torch.linalg.solve for each tetrahedron matrix T (avoiding explicit loops)
    T_non_singular = (
        T + torch.eye(3, device=T.device, dtype=T.dtype) * 1e-7
    )  # TODO why sometimes singular?
    bary_coords = torch.matmul(
        torch.linalg.inv(T_non_singular), relative_points.permute(1, 2, 0)
    ).permute(2, 0, 1)

    # Add the first barycentric coordinate for v0, which is 1 - sum of the other coordinates
    w0 = 1 - bary_coords.sum(
        dim=-1, keepdim=True
    )  # Shape: [N_points, N_tetrahedrons, 1]
    bary_coords = torch.cat(
        [w0, bary_coords], dim=-1
    )  # Shape: [N_points, N_tetrahedrons, 4]

    return bary_coords


def point_to_plane_distance(points, v0, v1, v2):
    # Compute the normal of the triangle
    normals = torch.cross(v1 - v0, v2 - v0, dim=1)
    normals = normals / torch.norm(normals, dim=1, keepdim=True)

    # Compute the vector from v0 to the points
    v0_to_points = points[:, None, :] - v0  # Expand v0 for broadcasting

    # Compute dot product between vectors to points and the normal
    distances = torch.abs(torch.einsum("ijk,ij->ik", v0_to_points, normals))

    return distances


def closest_point_on_segment(points, v0, v1):
    # Vector from v0 to v1
    v0_to_v1 = v1 - v0
    v0_to_v1_norm = torch.sum(v0_to_v1**2, dim=1, keepdim=True)

    # Vector from v0 to points
    v0_to_points = points[:, None, :] - v0

    # Compute projection factor t, and clamp between 0 and 1
    t = torch.einsum("ijk,ij->ik", v0_to_points, v0_to_v1) / v0_to_v1_norm
    t = torch.clamp(t, 0, 1)

    # Compute closest points on segment
    closest_points = v0 + t.unsqueeze(2) * v0_to_v1.unsqueeze(0)

    return closest_points


def point_to_edge_distance(points, v0, v1):
    closest_points = closest_point_on_segment(points, v0, v1)
    distances = torch.norm(points[:, None, :] - closest_points, dim=2)
    return distances


def closest_distance_to_tetrahedron(tetra_vertices, points):
    v0, v1, v2, v3 = [tetra_vertices[i, :] for i in range(4)]

    # Define the four faces of the tetrahedron
    f1 = torch.stack([v1, v2, v3], dim=0)
    f2 = torch.stack([v0, v2, v3], dim=0)
    f3 = torch.stack([v0, v1, v3], dim=0)
    f4 = torch.stack([v0, v1, v2], dim=0)
    faces = torch.stack([f1, f2, f3, f4], dim=0)
    e1 = torch.stack([v0, v2], dim=0)
    e2 = torch.stack([v0, v2], dim=0)
    e3 = torch.stack([v0, v3], dim=0)
    e4 = torch.stack([v1, v2], dim=0)
    e5 = torch.stack([v1, v3], dim=0)
    e6 = torch.stack([v2, v3], dim=0)
    edges = torch.stack([e1, e2, e3, e4, e5, e6], dim=0)
    vertices = torch.stack([v0, v1, v2, v3], dim=0)

    # Compute distances to faces (plane distances)
    face_distances = torch.stack(
        [point_to_plane_distance(points, *face) for face in faces]
    )

    # Compute distances to edges (segment distances)
    edge_distances = torch.stack(
        [point_to_edge_distance(points, *edge) for edge in edges]
    )

    # Compute distances to vertices
    vertex_distances = torch.norm(points[:, None, :] - vertices, dim=2)

    # Combine and get minimum distances for each point
    min_distances = torch.min(
        torch.cat(
            [face_distances, edge_distances, vertex_distances.unsqueeze(0)], dim=0
        ),
        dim=0,
    ).values

    return min_distances


# Step 3: Compute the weight of each point with respect to each tetrahedron
def tetrahedron_weights(point_cloud, tetrahedrons, inside_tetrahedrons: torch.Tensor):
    """
    Compute the weights of each point based on its proximity to the tetrahedrons.

    Args:
        point_cloud: Tensor of shape [N_points, 3]
        tetrahedrons: Tensor of shape [N_tetrahedrons, 4, 3]

    Returns:
        Weights: Tensor of shape [N_points, N_tetrahedrons]
    """
    centroids = torch.mean(tetrahedrons, dim=1)  # Shape: [N_tetrahedrons, 3]
    distances = torch.cdist(point_cloud, centroids)  # Shape: [N_points, N_tetrahedrons]
    min_dists = distances.min(dim=1).values  # Minimum distance to any tetrahedron
    weights = min_dists[:, None] / (distances + 1e-6)  # Inverse distance weight
    weights[~inside_tetrahedrons] = 0
    weights /= (
        weights.sum(dim=1, keepdim=True) + 1e-9
    )  # Normalize so weights sum to 1 per point

    return weights


def bary_coords_in_convex_hull(bary_coords, thresh=1e-6):
    """
    Check if the barycentric coordinates are inside the convex hull.

    bary_coords: Tensor of shape [N_points, N_tetrahedrons, 4]

    Returns:
    - Boolean tensor: True if the barycentric coordinates are inside the convex hull, False otherwise.
    """
    # Check if the point is inside the convex hull (all barycentric coordinates are non-negative)
    inside_tetrahedrons = (bary_coords >= -thresh).all(dim=-1) & (
        torch.abs(bary_coords.sum(dim=-1) - 1) < thresh
    )
    is_in_convex_hull = torch.any(inside_tetrahedrons, dim=1)

    return is_in_convex_hull, inside_tetrahedrons


def points_in_convex_hull(point_cloud, tetrahedrons, thresh=1e-6):
    """
    Check if each point in the point cloud is inside the convex hull defined by the tetrahedrons.

    point_cloud: Tensor of shape [N_points, 3]
    tetrahedrons: Tensor of shape [N_tetrahedrons, 4, 3]

    Returns:
    - Boolean tensor: True if the point is inside the convex hull, False otherwise.
    """
    # Compute the barycentric coordinates for all points with respect to all tetrahedrons
    bary_coords = barycentric_coordinates(
        point_cloud, tetrahedrons
    )  # Shape: [N_points, N_tetrahedrons, 4]

    # Check if the point is inside the convex hull (all barycentric coordinates are non-negative)
    is_in_convex_hull, _ = bary_coords_in_convex_hull(bary_coords, thresh=thresh)

    return bary_coords, is_in_convex_hull


def transform_pc(bary_coords, tet_weights, new_tetrahedrons):
    """
    Compute the new point cloud based on the barycentric coordinates and tetrahedron weights.
    Args:
        bary_coords: Tensor of shape [N_points, N_tetrahedrons, 4]
        tet_weights: Tensor of shape [N_points, N_tetrahedrons]
        new_tetrahedrons: Tensor of shape [N_tetrahedrons, 4, 3]
    """
    transformed_points = torch.einsum(
        "ab,abc,...bcd->...ad", tet_weights, bary_coords, new_tetrahedrons
    )
    return transformed_points


# Step 4: Apply the transformation for a point cloud with multiple tetrahedrons
def apply_tetrahedrons_to_point_cloud(
    point_cloud, tetrahedrons, new_tetrahedrons, thresh=1e-6
):
    """
    Apply the weighted transformation based on all tetrahedrons.

    point_cloud: Tensor of shape [N_points, 3]
    tetrahedrons: Tensor of shape [N_tetrahedrons, 4, 3]
    new_tetrahedrons: Tensor of shape [N_tetrahedrons, 4, 3]

    Returns:
    - Transformed point cloud: Tensor of shape [N_points, 3]
    """
    # Compute barycentric coordinates for all points with respect to all tetrahedrons
    bary_coords = barycentric_coordinates(
        point_cloud, tetrahedrons
    )  # Shape: [N_points, N_tetrahedrons, 4]

    is_in_convex_hull, inside_tetrahedrons = bary_coords_in_convex_hull(
        bary_coords, thresh=thresh
    )

    assert (
        torch.sum(is_in_convex_hull) > 0
    ), "No points are inside the convex hull of the tetrahedrons."

    # Compute weights for each point based on proximity to the tetrahedrons
    weights = tetrahedron_weights(
        point_cloud, tetrahedrons, inside_tetrahedrons
    )  # Shape: [N_points, N_tetrahedrons]

    # Use proper broadcasting to apply the barycentric coordinates and new tetrahedron positions
    # The correct einsum notation here should be 'npt,ntr->npr', and then sum over 'p'
    # transformed_points = bary_coords * new_tetrahedrons.unsqueeze(0)  # Shape: [N_points, N_tetrahedrons, 4, 3]
    transformed_points = torch.einsum(
        "ab,abc,bcd->ad", weights, bary_coords, new_tetrahedrons
    )

    # transformed point might no longer be in the convex hull, so we check again
    # bary_coords = barycentric_coordinates(transformed_points, new_tetrahedrons)
    # is_in_convex_hull, inside_tetrahedrons = bary_coords_in_convex_hull(
    #     bary_coords, thresh=thresh
    # )

    return transformed_points, is_in_convex_hull


# Step 5: Main function
def transform_point_cloud(
    skeleton_points, new_skeleton_points, point_cloud, thresh=1e-6
):
    """
    Main function to parametrize the point cloud based on tetrahedrons and transform it.

    skeleton_points: Tensor of shape [N_skeleton, 3]
    new_skeleton_points: Tensor of shape [N_skeleton, 3]
    point_cloud: Tensor of shape [N_points, 3]

    Returns:
    - Transformed point cloud: Tensor of shape [N_points, 3]
    """
    # Generate all tetrahedrons from the original skeleton points
    _, tetrahedrons = generate_tetrahedrons(skeleton_points)

    # Generate the corresponding new tetrahedrons from the new skeleton points
    _, new_tetrahedrons = generate_tetrahedrons(new_skeleton_points)

    # Apply the transformation to the point cloud
    # NOTE this part is heavy on computation
    transformed_cloud, is_in_convex_hull = apply_tetrahedrons_to_point_cloud(
        point_cloud, tetrahedrons, new_tetrahedrons, thresh=thresh
    )

    return transformed_cloud, is_in_convex_hull
