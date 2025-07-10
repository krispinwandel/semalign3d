"""Post-process correspondences via window-softmax"""

import torch
import numpy as np


def softmax_with_temperature(x, beta, d=1):
    """SFNet: Learning Object-aware Semantic Flow (Lee et al.)"""
    M, _ = x.max(dim=d, keepdim=True)
    x = x - M  # subtract maximum value for stability
    exp_x = torch.exp(x / beta)
    exp_x_sum = exp_x.sum(dim=d, keepdim=True)
    return exp_x / exp_x_sum


def soft_argmax(corr, beta=0.02):
    """SFNet: Learning Object-aware Semantic Flow (Lee et al.)"""
    # input shape : (B, H_t * W_t, H_s , W_s) e.g., (B, 32*32, 32, 32)
    b, htwt, h, w = corr.size()
    ht, wt = int(np.sqrt(htwt)), int(np.sqrt(htwt))
    x_normal = np.linspace(-1, 1, w)  # ?
    x_normal = torch.tensor(x_normal, device=corr.device).float()
    y_normal = np.linspace(-1, 1, h)  # ?
    y_normal = torch.tensor(y_normal, device=corr.device).float()

    corr = softmax_with_temperature(corr, beta=beta, d=1)  # (B, H_t * W_t, H_s , W_s)
    corr = corr.view(-1, ht, wt, h, w)  # (target hxw) x (source hxw)

    grid_x = corr.sum(dim=1, keepdim=False)  # marginalize to x-coord.
    x_normal = x_normal.expand(b, w)  # shape (b, w)
    x_normal = x_normal.view(b, w, 1, 1)  # shape (b, w, 1, 1)
    grid_x = (grid_x * x_normal).sum(dim=1, keepdim=True)  # b x 1 x h x w

    grid_y = corr.sum(dim=2, keepdim=False)  # marginalize to y-coord.
    y_normal = y_normal.expand(b, h)
    y_normal = y_normal.view(b, h, 1, 1)
    grid_y = (grid_y * y_normal).sum(dim=1, keepdim=True)  # b x 1 x h x w
    return grid_x, grid_y


def unnormalise_and_convert_mapping_to_flow(map):
    # here map is normalised to -1;1
    # we put it back to 0,W-1, then convert it to flow
    B, C, H, W = map.size()
    mapping = torch.zeros_like(map)
    # mesh grid
    mapping[:, 0, :, :] = (
        (map[:, 0, :, :].float().clone() + 1) * (W - 1) / 2.0
    )  # unormalise
    mapping[:, 1, :, :] = (
        (map[:, 1, :, :].float().clone() + 1) * (H - 1) / 2.0
    )  # unormalise
    flow = mapping
    return flow


# TODO flow is a confusing name here, it is just window soft argmax
def compute_flow(src_img_embd, trg_img_embd, flow_window=7):
    """
    Note:
        Let p^* = (x^*, y^*) be the maximum correlation point.
        Instead of using p^* directly, we take a weighted average of all points in a window around p^*.
        The weights are computed using softmax.
    Args:
        src_img_embd: (B, C, num_patches, num_patches)
        trg_img_embd: (B, C, num_patches, num_patches)
    Returns:
        flow: (B, num_patches, num_patches, 2) - mappping from src to trg
    """
    b, c, num_patches = src_img_embd.shape[:3]
    np2 = num_patches * num_patches
    src_img_embd_flat = src_img_embd.reshape(1, c, np2)
    trg_img_embd_flat = trg_img_embd.reshape(1, c, np2)
    corr = torch.bmm(
        src_img_embd_flat.permute(0, 2, 1), trg_img_embd_flat
    )  # (b, num_patches*num_patches, num_patches*num_patches)

    max_index_flatten = torch.argmax(corr, dim=-1)
    max_index_x = max_index_flatten % num_patches  # (b, num_patches * num_patches, )
    max_index_y = max_index_flatten // num_patches  # (b, num_patches * num_patches, )
    corr = corr.view(
        b, np2, num_patches, num_patches
    )  # (b, num_patches * num_patches, num_patches, num_patches)

    # Prepare offsets
    offset_range = torch.arange(-flow_window, flow_window + 1, device=corr.device)
    offset_x, offset_y = torch.meshgrid(offset_range, offset_range, indexing="ij")
    offset_x, offset_y = offset_x.flatten(), offset_y.flatten()  # (window_size^2, )

    # Compute window mask without loops
    window_positions_x = (max_index_x[:, :, None] + offset_x[None, None, :]).clamp(
        0, num_patches - 1
    )  # (b, num_patches * num_patches, window_size^2)
    window_positions_y = (max_index_y[:, :, None] + offset_y[None, None, :]).clamp(
        0, num_patches - 1
    )  # (b, num_patches * num_patches, window_size^2)

    # Create indices for gathering values
    batch_indices = torch.arange(b, device=corr.device)[:, None, None]  # (b, 1, 1)
    src_indices = torch.arange(np2, device=corr.device)[
        None, :, None
    ]  # (1, num_patches * num_patches, 1)

    # Using advanced indexing to create the window mask
    window_mask = torch.zeros_like(
        corr
    )  # (b, num_patches * num_patches, num_patches, num_patches)
    window_mask[batch_indices, src_indices, window_positions_y, window_positions_x] = 1

    # Apply window mask
    corr = corr * window_mask  # (num_patches * num_patches, num_patches, num_patches)

    x = corr.view(b, num_patches, num_patches, np2)
    grid_x, grid_y = soft_argmax(x.permute(0, 3, 1, 2))
    x = torch.cat((grid_x, grid_y), dim=1)  # (B, 2, H, W)
    x = unnormalise_and_convert_mapping_to_flow(x)  # (B, 2, H, W)
    return x.permute(0, 2, 3, 1)
