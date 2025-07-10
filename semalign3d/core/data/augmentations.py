import torch
from semalign3d.core import data_classes


def generate_noisy_gt_kpts_data(
    gt_kpts_data: data_classes.GtKptsData, n_samples_per_img=10, noise_rate=0.3
):
    n_imgs = gt_kpts_data.xyz.shape[0]
    noisy_xyz = gt_kpts_data.xyz.repeat_interleave(n_samples_per_img, dim=0)
    vertex_mask = gt_kpts_data.vertex_mask.repeat_interleave(n_samples_per_img, dim=0)
    for i in range(n_imgs):
        kpts_i = gt_kpts_data.xyz[i][gt_kpts_data.vertex_mask[i]]
        # print(kpts_i)
        edge_combinations = torch.combinations(torch.arange(kpts_i.shape[0]), 2)
        edges = kpts_i[edge_combinations[:, 0]] - kpts_i[edge_combinations[:, 1]]
        edge_lenghts = torch.norm(edges, dim=1)
        edge_lenghts[torch.isnan(edge_lenghts)] = torch.inf
        # edge_lenghts[edge_lenghts < 1e-3] = torch.inf
        min_edge_legnth = edge_lenghts.min()
        # print(min_edge_legnth)
        # max_edge_length = edge_lenghts.max()
        for j in range(n_samples_per_img):
            noise = torch.randn_like(kpts_i) * min_edge_legnth * noise_rate
            xyz = kpts_i + noise
            noisy_xyz[i * n_samples_per_img + j, gt_kpts_data.vertex_mask[i]] = xyz
    return data_classes.GtKptsData(xyz=noisy_xyz, vertex_mask=vertex_mask)
