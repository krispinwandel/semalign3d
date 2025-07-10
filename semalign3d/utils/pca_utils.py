"""PCA utilities for keypoint features"""

import torch
from tqdm import tqdm
from semalign3d.utils import image_processing
from semalign3d.core import data_classes


def get_feature_mask(seg_mask: torch.Tensor, embd_size: int, threshold=0.2):
    h, w = seg_mask.shape
    pad_h, pad_w = image_processing.get_pad_sizes_from_img_shape(h, w)
    patch_size = max(h, w) / embd_size
    ft_mask = torch.zeros((embd_size, embd_size), dtype=torch.bool)
    for x in range(embd_size):
        for y in range(embd_size):
            x_orig, y_orig = (
                int(x * patch_size) - pad_w // 2,
                int(y * patch_size) - pad_h // 2,
            )
            x_orig_min, y_orig_min = max(x_orig, 0), max(y_orig, 0)
            x_orig_max, y_orig_max = int(x_orig + patch_size), int(y_orig + patch_size)
            x_orig_max, y_orig_max = max(x_orig_max, 0), max(y_orig_max, 0)
            if (
                torch.sum(seg_mask[y_orig_min:y_orig_max, x_orig_min:x_orig_max])
                / (patch_size**2)
                > threshold
            ):
                ft_mask[y, x] = True
    return ft_mask


def extract_features_on_mask(processed_data: data_classes.SpairProcessedData):
    fts = []
    for idx in tqdm(range(len(processed_data.seg_masks))):
        seg_mask_torch = torch.from_numpy(processed_data.seg_masks[idx][2])
        ft_mask = get_feature_mask(seg_mask=seg_mask_torch, embd_size=60, threshold=0.2)
        # TODO should we really used normalized features? => apparently does not really matter (inspect further)
        fts.append(processed_data.img_embds_hat[idx][:, ft_mask].T)
    fts = torch.cat(fts, dim=0)
    return fts


def pca(X, k=None):
    """
    Perform PCA on the input data.

    Args:
        X (torch.Tensor): Input tensor of shape (n, d), where n is the number of samples and d is the feature dimension.
        k (int, optional): Number of principal components to keep. If None, all components are kept.

    Returns:
        torch.Tensor: PCA components of shape (n, k) if k is provided, otherwise (n, d).
        torch.Tensor: The eigenvalues of the covariance matrix.
        torch.Tensor: The eigenvectors (principal components).
    """
    # Step 1: Normalize the data
    X_mean = torch.mean(X, dim=0)
    X_std = torch.std(X, dim=0)
    X_hat = (X - X_mean) / X_std  # Normalize the data

    # Step 2: Compute the covariance matrix
    covariance_matrix = torch.mm(X_hat.T, X_hat) / (X_hat.shape[0] - 1)

    # Step 3: Compute the eigenvalues and eigenvectors of the covariance matrix using torch.linalg.eigh
    eigenvalues, eigenvectors = torch.linalg.eigh(covariance_matrix)

    # Step 4: Sort the eigenvalues and eigenvectors in descending order
    sorted_indices = torch.argsort(eigenvalues, descending=True)
    sorted_eigenvalues = eigenvalues[sorted_indices]
    sorted_eigenvectors = eigenvectors[:, sorted_indices]

    # Step 5: Project the data onto the principal components
    if k is not None:
        sorted_eigenvectors = sorted_eigenvectors[:, :k]

    X_pca = torch.mm(X_hat, sorted_eigenvectors)

    return X_pca, X_mean, X_std, sorted_eigenvalues, sorted_eigenvectors


def apply_pca(X_new, pca_eigenvectors, original_mean, original_std):
    """
    Apply PCA to new data using previously computed eigenvectors (principal components) and the original mean & std.

    Args:
        X_new (torch.Tensor): New input tensor of shape (n, d), where n is the number of samples and d is the feature dimension.
        pca_eigenvectors (torch.Tensor): Eigenvectors (principal components) from the previously computed PCA, of shape (d, k), where k is the number of principal components.
        original_mean (torch.Tensor): The mean of the original data (used when computing PCA), of shape (d,).
        original_std (torch.Tensor): The standard deviation of the original data (used when computing PCA), of shape (d,).

    Returns:
        torch.Tensor: Transformed data of shape (n, k) projected onto the principal components.
    """
    # Step 1: Normalize the new data using the original mean & std
    X_new_hat = (X_new - original_mean) / original_std

    # Step 2: Project the new data onto the principal components
    X_new_pca = torch.mm(X_new_hat, pca_eigenvectors)

    return X_new_pca


def compute_keypoint_features(
    preprocessed_data_train: data_classes.SpairProcessedData,
    pca_eigenvectors: torch.Tensor,
    original_mean: torch.Tensor,
    n_max_kpts=30,
):
    """Compute keypoint features"""
    # average keypoint embeddings
    kpt_features_pca_avg_train = []
    kpt_features_pca_train = []
    kpt_fts_mean = []
    kpt_fts_evs = []
    kpt_fts_pca_mean = []
    C = pca_eigenvectors.shape[1]

    for kpt_idx in tqdm(range(n_max_kpts)):
        kpt_features = preprocessed_data_train.kpt_idx_to_kpt_embds_hat[kpt_idx]

        # kpt_features is only None if keypoint label does not exist for object category
        if kpt_features is None:
            # kpt does not exist
            kpt_features_pca_avg_train.append(torch.zeros((1, C), dtype=torch.float32))
            kpt_fts_mean.append(torch.zeros((C,), dtype=torch.float32))
            kpt_fts_evs.append(torch.zeros((C, C), dtype=torch.float32))
            kpt_fts_pca_mean.append(torch.zeros((C,), dtype=torch.float32))
            continue
        X_pca, X_mean, sorted_eigenvalues, sorted_eigenvectors = pca(kpt_features)
        kpt_fts_mean.append(X_mean)
        kpt_fts_evs.append(sorted_eigenvectors)
        kpt_fts_pca_mean.append(torch.mean(X_pca, dim=0))
        kpt_features_pca = apply_pca(kpt_features, pca_eigenvectors, original_mean)
        kpt_features_pca_avg = torch.mean(
            kpt_features_pca, dim=0, keepdim=True
        )  # (1, C)
        kpt_features_pca_avg_train.append(kpt_features_pca_avg)
        kpt_features_pca_train.append(kpt_features_pca)

    kpt_features_pca_avg_train = torch.cat(kpt_features_pca_avg_train, dim=0)
    return (
        kpt_features_pca_avg_train,
        kpt_features_pca_train,
        kpt_fts_mean,
        kpt_fts_evs,
        kpt_fts_pca_mean,
    )


def tensor_to_rgb(tensor: torch.Tensor, tensor_min=None, tensor_max=None):
    # Ensure the tensor is of shape (n, 3)
    if tensor.shape[1] != 3:
        raise ValueError("Input tensor must have shape (n, 3)")

    if tensor_min is None:
        tensor_min = tensor.min()

    # Normalize the tensor values to the range [0, 255]
    tensor = tensor - tensor_min
    if tensor_max is None:
        tensor_max = tensor.max()
    tensor = tensor / tensor_max
    tensor = tensor * 255.0
    tensor = tensor.clamp(0, 255)

    # Convert to integer type
    tensor = tensor.to(torch.uint8)

    return tensor, tensor_min, tensor_max
