import cudf
import cuml
import torch


def compute_k_nearest_neighbours(X, k=5):
    """
    Args:
        - X: torch.Tensor of shape [N, D]
        - k: int
    """

    dists = torch.cdist(X, X)
    on_diagonal = dists < 1e-3
    dists = torch.where(on_diagonal, torch.inf, dists)
    k_nearest_neighbours = torch.topk(dists, k=k, largest=False)
    topk_dists = k_nearest_neighbours.values  # Shape: [N, k]
    topk_indices = k_nearest_neighbours.indices  # Shape: [N, k]
    topk_weights = torch.min(topk_dists, dim=1, keepdim=True).values / topk_dists
    topk_weights = topk_weights / topk_weights.sum(dim=1, keepdim=True)
    return topk_indices, topk_dists, topk_weights


def kmeans_torch(X: torch.Tensor, n_clusters, random_state=42):
    """
    Cluster features using KMeans.

    Args:
        X (torch.Tensor): Input features of shape (n_samples, n_features).
        n_clusters (int): Number of clusters.
        random_state (int): Random seed for reproducibility.

    Returns:
        torch.Tensor: Cluster labels for each sample.
        torch.Tensor: Cluster centers.
    """
    data_gpu = cudf.DataFrame(X.cpu().numpy())
    kmeans_gpu = cuml.KMeans(n_clusters=n_clusters, random_state=random_state)
    kmeans_gpu.fit(data_gpu)
    cluster_labels_gpu = kmeans_gpu.labels_
    cluster_centers_gpu = kmeans_gpu.cluster_centers_
    cluster_centers = torch.tensor(cluster_centers_gpu.to_cupy())
    cluster_labels = torch.tensor(cluster_labels_gpu.to_cupy())
    return {
        "cluster_labels": cluster_labels,
        "cluster_centers": cluster_centers,
    }
