import torch
import numpy as np
import pyquaternion as pyq
from typing import Optional


def reposition_kpt_xyz_obj(
    kpt_xyz_obj: torch.Tensor,
    kpt_xyz_img: torch.Tensor,
    used_kpts_mask: Optional[torch.Tensor] = None,
):
    if used_kpts_mask is None:
        used_kpts_mask = torch.ones(kpt_xyz_obj.shape[0], dtype=torch.bool)
    xyz_repositioned = kpt_xyz_obj.clone()
    # center
    xyz_repositioned -= torch.mean(xyz_repositioned[used_kpts_mask], dim=0)
    # scale
    kpt_max_dist = torch.max(torch.cdist(kpt_xyz_img, kpt_xyz_img).reshape(-1))
    kpt_max_dist_obj = torch.max(
        torch.cdist(
            xyz_repositioned[used_kpts_mask], xyz_repositioned[used_kpts_mask]
        ).reshape(-1)
    )
    # TODO 3.0 should be a parameter
    # TODO allow for double distance
    xyz_repositioned *= kpt_max_dist / kpt_max_dist_obj * 3.0
    # re-center
    xyz_repositioned += torch.mean(kpt_xyz_img, dim=0)
    return xyz_repositioned


def generate_rotation_samples(
    xyz: torch.Tensor, n_samples=100, used_xyz: Optional[torch.Tensor] = None
) -> torch.Tensor:

    if used_xyz is None:
        used_xyz = torch.ones(xyz.shape[0], dtype=torch.bool)

    xyz_mean = torch.mean(xyz[used_xyz], dim=0, keepdim=True)
    xyz = xyz - xyz_mean

    # generate random rotations
    rotations = []
    for i in range(n_samples):
        q = pyq.Quaternion.random()
        rotations.append(q.rotation_matrix)
    rotations = torch.from_numpy(np.stack(rotations)).type(xyz.dtype)
    # generate initial configurations
    xyz_samples = xyz[None, :, :].repeat(n_samples, 1, 1)
    xyz_samples = torch.bmm(rotations, xyz_samples.permute(0, 2, 1)).permute(0, 2, 1)

    # re-center
    xyz_samples += xyz_mean[None, :, :]
    return xyz_samples
