import numpy as np
import torch
from semalign3d.utils import camera_intrinsics


def project_to_img_plane(xyz_world: torch.Tensor, K: torch.Tensor):
    """
    Args:
        xyz_world: (n, 3)
        K: (4, 4) is the calibration matrix
            (we assume that the camera is located at the origin and looks along the z-axis.
            That is, T_cam = [I|0] and R_cam = I)
    Out:
        xy_img: (n, 2)
    """
    xy_img = torch.matmul(K, xyz_world.T).T
    xy_img = xy_img[:, :2] / (xy_img[:, 2][:, None] + 1e-9)
    return xy_img


def project_to_img_plane_with_focal_length(
    xyz_coords: torch.Tensor,
    img_shape: torch.Tensor,
    img_focal_length_inv: torch.Tensor,
    use_clamp=True,
):
    img_h, img_w = img_shape
    img_K = camera_intrinsics.get_normalized_calibration_matrix_np(
        img_h.item(), img_w.item(), f=1 / img_focal_length_inv.item()
    )
    img_xy_pred = project_to_img_plane(xyz_coords, K=torch.from_numpy(img_K).float())
    img_xy_pred = img_xy_pred * torch.max(img_shape)
    if use_clamp:
        img_xy_pred[:, 0] = torch.clamp(
            img_xy_pred[:, 0], torch.tensor([0]), img_shape[1] - 1
        )
        img_xy_pred[:, 1] = torch.clamp(
            img_xy_pred[:, 1], torch.tensor([0]), img_shape[0] - 1
        )
    return img_xy_pred


# @torch.compile()
def project_to_img_plane_batched(xyz_world: torch.Tensor, K: torch.Tensor):
    """
    Args:
        xyz_world: (B, n, 3)
        K: (B, 3, 3) is the calibration matrix
            (we assume that the camera is located at the origin and looks along the z-axis.
            That is, T_cam = [I|0] and R_cam = I)
    Out:
        xy_img: (B, n, 2)
    """
    xy_img_homogeneous = torch.matmul(xyz_world, K.transpose(1, 2))
    # NOTE very important to use torch.split here, otherwise compilation fails for the backward pass
    xy, z = torch.split(xy_img_homogeneous, [2, 1], dim=-1)
    xy_img = xy / (z + 1e-9)
    return xy_img


def project_to_img_plane_from_K_normalized(
    xyz_world: torch.Tensor, K: torch.Tensor, use_clamp=True
):
    """
    Args:
        xyz_world: (n, 3)
        K: (3, 3) is the calibration matrix
            (we assume that the camera is located at the origin and looks along the z-axis.
            That is, T_cam = [I|0] and R_cam = I)
    Out:
        xy_img: (n, 2)
    """
    xy_img = torch.matmul(K, xyz_world.T).T
    xy_img = xy_img[:, :2] / (xy_img[:, 2][:, None] + 1e-9)
    if use_clamp:
        xy_img[:, 0] = torch.clamp(xy_img[:, 0], min=0)
        xy_img[:, 1] = torch.clamp(xy_img[:, 1], min=0)
    return xy_img


def back_project_to_world(
    xy_img: torch.Tensor, depth_values: torch.Tensor, K_inv: torch.Tensor
):
    """
    Args:
        xy_img: (n, 2)
        depth_values: (n)
        K_inv: (3, 3) is the inverse of the calibration matrix K[:3,:3]
    Out:
        xyz_world: (n, 3)
    """
    xyz_cam_plane = torch.cat([xy_img, depth_values.unsqueeze(1)], dim=1)
    xyz_world = torch.matmul(K_inv, xyz_cam_plane.T)
    return xyz_world.T


def back_project_depth_image(depth_img: np.ndarray, focal_length=0.3):
    """
    Args:
        depth_img: (h, w)
    """
    img_h, img_w = depth_img.shape
    K = camera_intrinsics.get_normalized_calibration_matrix_torch(
        img_h, img_w, f=focal_length
    )
    # T = np.eye(4)
    # pixel_size = 20*0.036 / max(img_h,img_w)
    x, y = np.meshgrid(np.arange(img_w), np.arange(img_h))
    x = x.astype(float) / max(img_w, img_h)
    y = y.astype(float) / max(img_w, img_h)
    # x *= pixel_size
    # y *= pixel_size
    xyz_cam_plane = np.stack([x * depth_img, y * depth_img, depth_img])  # (3,h,w)
    xyz_cam_plane = xyz_cam_plane.reshape((3, img_h * img_w))
    xyz_world = (np.linalg.inv(K[:3, :3]) @ xyz_cam_plane).T
    xyz_world = xyz_world.reshape((img_h, img_w, 3))
    # print(np.linalg.inv(K[:3,:3]))
    return xyz_world


def back_project_kpts_from_K_inv_normalized(
    K_inv_normalized: torch.Tensor,
    kpt_xy_normalized: torch.Tensor,
    kpt_depth: torch.Tensor,
):
    """
    Args:
        K_inv_normalized: (n_imgs, 3, 3)
        kpt_xy_normalized: (n_imgs, n_max_kpts, 2)
        kpt_depth: (n_imgs, n_max_kpts)
    """
    x, y = torch.split(kpt_xy_normalized, [1, 1], dim=-1)
    x = x * kpt_depth.unsqueeze(-1)
    y = y * kpt_depth.unsqueeze(-1)
    z = kpt_depth.unsqueeze(-1)
    kpt_xyz = torch.cat([x, y, z], dim=-1)  # (n_imgs, n_max_kpts, 3)
    kpt_xyz = kpt_xyz.unsqueeze(-1)  # (n_imgs, n_max_kpts, 3, 1)
    kpt_xyz = torch.matmul(
        K_inv_normalized.unsqueeze(1), kpt_xyz
    )  # (n_imgs, n_max_kpts, 3, 1)
    kpt_xyz = kpt_xyz.squeeze(-1)  # (n_imgs, n_max_kpts, 3)
    return kpt_xyz


def back_project_kpts(
    focal_lenghts_inv: torch.Tensor,
    img_shapes: torch.Tensor,
    kpt_xy_normalized: torch.Tensor,
    kpt_depth: torch.Tensor,
):
    """
    Args:
        focal_lenghts_inv: (n_imgs,) 1 / focal_length
        img_shapes: (n_imgs, 2). img_shapes[:,0] = img height (y), img_shapes[:,1] = img width (x)
        kpt_xy_normalized: (n_imgs, n_max_kpts, 2)
        kpt_depth: (n_imgs, n_max_kpts)
    """
    n_imgs = focal_lenghts_inv.shape[0]
    focal_lenghts_inv_abs = torch.abs(focal_lenghts_inv)

    # TODO K_inv construction should move to camera_intrinsics.py
    # 1) construct K_inv for each image in parallel and analytically
    # NOTE K_inv = [[1/f, 0, -px/f],[0, 1/f, -py/f],[0, 0, 1]]
    K_inv = torch.zeros(
        (n_imgs, 3, 3), device=focal_lenghts_inv.device, dtype=focal_lenghts_inv.dtype
    )
    K_inv[:, 0, 0] = focal_lenghts_inv_abs
    K_inv[:, 1, 1] = focal_lenghts_inv_abs
    K_inv[:, 2, 2] = 1
    img_max_size = torch.max(img_shapes, dim=1).values
    px = img_shapes[:, 1] / img_max_size / 2
    py = img_shapes[:, 0] / img_max_size / 2
    K_inv[:, 0, 2] = -px * focal_lenghts_inv_abs
    K_inv[:, 1, 2] = -py * focal_lenghts_inv_abs

    # 2) reproject kpt_xy to world coordinates
    return back_project_kpts_from_K_inv_normalized(
        K_inv_normalized=K_inv,
        kpt_xy_normalized=kpt_xy_normalized,
        kpt_depth=kpt_depth,
    )
