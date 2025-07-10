import torch


def get_cube_grid(center: torch.Tensor, size: float, n_points: int):
    """
    Get a grid of points in a cube centered at center with size
    """
    x = torch.linspace(
        center[0].item() - size / 2, center[0].item() + size / 2, n_points
    )
    y = torch.linspace(
        center[1].item() - size / 2, center[1].item() + size / 2, n_points
    )
    z = torch.linspace(
        center[2].item() - size / 2, center[2].item() + size / 2, n_points
    )
    x, y, z = torch.meshgrid(x, y, z)
    return torch.stack([x, y, z], dim=-1).reshape(-1, 3)


def create_line_pts(
    K_inv_normalized: torch.Tensor,
    line_pts_xy_normalized: torch.Tensor,
):
    K_inv_00 = K_inv_normalized[0, 0]
    K_inv_11 = K_inv_normalized[1, 1]
    K_inv_02 = K_inv_normalized[0, 2]
    K_inv_12 = K_inv_normalized[1, 2]
    # depth is 1.0
    line_p1_x = K_inv_00 * line_pts_xy_normalized[:, 0] + 1.0 * K_inv_02
    line_p1_y = K_inv_11 * line_pts_xy_normalized[:, 1] + 1.0 * K_inv_12
    line_p1_z = torch.ones_like(line_p1_x)
    line_p2_x = line_p1_x * 0.4
    line_p2_y = line_p1_y * 0.4
    line_p2_z = line_p1_z * 0.4
    line_p1 = torch.stack([line_p1_x, line_p1_y, line_p1_z], dim=-1)  # (n_l, 3)
    line_p2 = torch.stack([line_p2_x, line_p2_y, line_p2_z], dim=-1)  # (n_l, 3)
    line_pts = torch.stack([line_p1, line_p2], dim=0)  # (2, n_l, 3)
    return line_pts
