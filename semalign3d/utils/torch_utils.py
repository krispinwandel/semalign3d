import numpy as np
import torch
from typing import Union, List, Tuple, Dict, TypeVar, Any
import os


def to_numpy(t: Any) -> Any:
    if isinstance(t, dict):
        for k, v in t.items():
            t[k] = to_numpy(v)
        return t
    elif isinstance(t, torch.Tensor):
        return t.detach().cpu().numpy()
    else:
        return t


def to_torch(t, device="cuda"):
    if isinstance(t, dict):
        for k, v in t.items():
            t[k] = to_torch(v, device)
        return t
    elif isinstance(t, np.ndarray):
        return torch.from_numpy(t).to(device)
    else:
        return t.to(device)


def to_device(
    t: Dict[str, Union[torch.Tensor, Dict[str, Any]]], device="cuda"
) -> Dict[str, Union[torch.Tensor, Dict[str, Any]]]:
    """
    Recursively move tensors in a dictionary to the specified device.

    Args:
        t (Dict[str, Union[torch.Tensor, Dict[str, Any]]]): The input dictionary containing tensors or nested dictionaries.
        device (str): The target device (default is "cuda").

    Returns:
        Dict[str, Union[torch.Tensor, Dict[str, Any]]]: The input dictionary with tensors moved to the specified device.
    """
    for k, v in t.items():
        if isinstance(v, torch.Tensor):
            t[k] = v.to(device)
        elif isinstance(v, dict):
            t[k] = to_device(v, device)
    return t


def clone_to_device(
    t: Dict[str, Union[torch.Tensor, Dict[str, Any]]], t_clone, device="cuda"
) -> Dict[str, Union[torch.Tensor, Dict[str, Any]]]:
    """
    Recursively clone tensors in a dictionary and move them to the specified device.

    Args:
        t (Dict[str, Union[torch.Tensor, Dict[str, Any]]]): The input dictionary containing tensors or nested dictionaries.
        t_clone (Dict[str, Union[torch.Tensor, Dict[str, Any]]]): The output dictionary to store cloned tensors.
        device (str): The target device (default is "cuda").

    Returns:
        Dict[str, Union[torch.Tensor, Dict[str, Any]]]: The output dictionary with cloned tensors moved to the specified device.
    """
    for k, v in t.items():
        if isinstance(v, torch.Tensor):
            t_clone[k] = v.clone().to(device)
        elif isinstance(v, dict):
            t_clone[k] = clone_to_device(v, {}, device)
        else:
            t_clone[k] = v
    return t_clone


def to_np_array(t: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    if isinstance(t, torch.Tensor):
        return t.detach().cpu().numpy()
    else:
        return t


def to_torch_tensor(
    t: Union[torch.Tensor, np.ndarray], device="cuda", dtype=torch.float32
) -> torch.Tensor:
    if isinstance(t, np.ndarray):
        return torch.tensor(t, device=device, dtype=dtype)
    else:
        return t.to(device)


def nanvar(tensor: torch.Tensor, dim: int):
    valid_elements = tensor[~torch.isnan(tensor)]
    if len(valid_elements) == 0:
        print("warning: all elements are NaN", valid_elements)
        return torch.tensor(float("nan"))
    mean = valid_elements.mean()
    return ((valid_elements - mean) ** 2).mean()


# Helper function to compute standard deviation, ignoring NaNs
def nanstd(tensor: torch.Tensor, dim: int):
    return nanvar(tensor, dim).sqrt()


def masked_mean(x, mask, dim=0):
    """
    Args:
        x: (d1,d2,d3)
        mask: (d1,d2,d3)
    """
    # n = x.shape[0]
    # mask = mask.view(n, -1)
    # x = x.view(n, -1)
    mask_sum = torch.sum(mask, dim=dim, keepdim=True)
    x_hat = x.clone()
    x_hat[~mask] = 0
    mask_sum[mask_sum == 0] = 1  # NOTE 0/1 = 0
    factor = mask / mask_sum
    x_mean = x_hat * factor
    x_mean = torch.sum(x_mean, dim=dim, keepdim=True)
    return x_mean.squeeze(dim)


def weighted_mean(x, weights, dim=0, keepdim=False):
    weights_sum = torch.sum(weights, dim=dim, keepdim=keepdim)
    return torch.sum(x * weights, dim=dim, keepdim=keepdim) / torch.clip(
        weights_sum, min=1
    )


def masked_var_mean(x, mask, dim=0):
    """
    Args:
        x: (d1,d2,d3)
        mask: (d1,d2,d3)
    """
    # n = x.shape[0]
    # mask = mask.view(n, -1)
    # x = x.view(n, -1)
    mask_sum = torch.sum(mask, dim=dim, keepdim=True)
    x_hat = x.clone()
    x_hat[~mask] = 0
    mask_sum[mask_sum == 0] = 1  # NOTE 0/1 = 0
    factor_avg = mask / mask_sum
    x_mean = x_hat * factor_avg
    x_mean = torch.sum(x_mean, dim=dim, keepdim=True)
    x_var = x_hat - x_mean
    x_var = x_var**2
    factor_var = mask / torch.clip(mask_sum - 1, min=1)
    x_var = x_var * factor_var
    x_var = torch.sum(x_var, dim=dim, keepdim=True)
    return x_var.squeeze(dim), x_mean.squeeze(dim)


def pseud_masked_var(x, x_mean, mask, dim=0):
    """
    Args:
        x: (d1,d2,d3)
        x_mean: (d1,d2,d3)
        mask: (d1,d2,d3)
    """
    x_hat = x - x_mean.unsqueeze(dim)
    var_hat = torch.max(torch.abs(x_hat * mask), dim=dim).values ** 2
    return var_hat.squeeze()


def unique_rows(x: torch.Tensor):
    assert len(x.shape) == 2
    x_np = to_np_array(x)
    _, row_indices_np = np.unique(x_np, axis=0, return_index=True)
    row_indices_torch = to_torch_tensor(row_indices_np, device=x.device).type(x.dtype)
    # NOTE the below is false
    # _, inverse_indices = torch.unique(x, return_inverse=True, dim=0)
    # row_indices = torch.unique(inverse_indices, sorted=True, return_inverse=False)
    return x[row_indices_torch, :], row_indices_torch


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def min_max_norm(ft, eps=1e-10):
    norms = torch.linalg.norm(ft, dim=1, keepdim=True)
    ft = ft / (norms + eps)
    return ft


def generate_is_not_nan_mask(x: torch.Tensor, dim=-1):
    """
    Args:
        x: (d1,d2,d3,...)
    """
    x_is_not_nan = torch.logical_not(torch.any(torch.isnan(x), dim=dim))
    return x_is_not_nan


def print_cuda_memory_usage():
    if torch.cuda.is_available():
        print(f"Allocated: {torch.cuda.memory_allocated() / 1024 ** 2} MB")
        print(f"Cached: {torch.cuda.memory_reserved() / 1024 ** 2} MB")
    else:
        print("CUDA is not available.")


# @torch.compile
# def dist2_fast(A,B):
#     # 1) compute squared norms
#     A2 = (A**2).sum(dim=2, keepdim=True)      # (B, N, 1)
#     B2 = (B**2).sum(dim=1).view(1, 1, B.shape[0])       # (1, 1, M)

#     # 2) batched GEMM for A·Bᵀ
#     #    torch.matmul will call cuBLAS under the hood
#     AB = torch.matmul(A, B.T)                 # (B, N, M)

#     # 3) assemble squared distances
#     dist2 = A2 + B2 - 2*AB                     # (B, N, M)
#     return dist2


# @torch.compile
# def dist2_fast(A,B):
#     # 1) compute squared norms
#     A2 = (A**2).sum(dim=-1, keepdim=True)      # (B, N, 1)
#     B2 = (B**2).sum(dim=-1, keepdim=True)      # (B, M, 1)

#     # 2) batched GEMM for A·Bᵀ
#     #    torch.matmul will call cuBLAS under the hood
#     AB = torch.matmul(A, B.permute(0, 2, 1))                 # (B, N, M)

#     # 3) assemble squared distances
#     dist2 = A2 + B2.permute(0, 2, 1) - 2*AB                     # (B, N, M)
#     return dist2
