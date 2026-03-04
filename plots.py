"""
Plotting utilities for K-Means clustering analysis.

Generates four types of figures for each dataset:
  1. Cluster visualization (PCA to 2D, colored by cluster) for k in [2, 5, 10, 20].
  2. Convergence: SSE per iteration for Alternate vs. Sklearn k-means.
  3. Reassigned: number of points reassigned per iteration (Alternate only).
  4. Runtime: time vs. k for Alternate vs. Sklearn.

All plots use KMeansAlternate from kmeans_alternate.py and sklearn.cluster.KMeans.
Figures are saved under plots/ with names like clusters_<dataset_name>.png.
"""

import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from kmeans_alternate import KMeansAlternate

# k values used across all plot types for consistency
K_VALUES = [2, 5, 10, 20]


def reduce_to_2d(data):
    """
    Reduce data to 2 dimensions using PCA for visualization.

    Args:
        data: Array of shape (n_samples, n_features).

    Returns:
        np.ndarray: Shape (n_samples, 2), first two principal components.
    """
    pca = PCA(n_components=2)
    return pca.fit_transform(data)


def plot_clusters(data, dataset_name):
    """
    Plot cluster assignments in 2D PCA space for k = 2, 5, 10, 20.

    Each subplot shows points colored by cluster label from KMeansAlternate.
    Saves to plots/clusters_<dataset_name>.png.

    Args:
        data: Array of shape (n_samples, n_features).
        dataset_name: Label for title and filename (e.g. 'Iris', 'AI_Index').
    """
    data_2d = reduce_to_2d(data)
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    fig.suptitle(f"Cluster Visualization — {dataset_name}")

    for i, k in enumerate(K_VALUES):
        model = KMeansAlternate(k=k)
        model.fit(data)

        axes[i].scatter(
            data_2d[:, 0],
            data_2d[:, 1],
            c=model.labels,
            cmap="tab10",
            s=20,
        )
        axes[i].set_title(f"k={k}")
        axes[i].set_xlabel("PC1")
        axes[i].set_ylabel("PC2")

    plt.tight_layout()
    plt.savefig(f"plots/clusters_{dataset_name}.png")
    plt.close()
    print(f"Saved cluster plot for {dataset_name}")


def plot_convergence(data, dataset_name):
    """
    Plot SSE vs. iteration for KMeansAlternate and Sklearn k-means (random init).

    For Sklearn we reconstruct the per-iteration SSE by fitting with
    max_iter=1, 2, ... up to the actual n_iter_. Saves to
    plots/convergence_<dataset_name>.png.

    Args:
        data: Array of shape (n_samples, n_features).
        dataset_name: Label for title and filename.
    """
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    fig.suptitle(f"Convergence Behavior — {dataset_name}")

    for i, k in enumerate(K_VALUES):
        alt = KMeansAlternate(k=k)
        alt.fit(data)

        sk = KMeans(n_clusters=k, random_state=42, n_init=1)
        sk.fit(data)
        # Rebuild Sklearn SSE per iteration by refitting with increasing max_iter
        sk_sse = []
        for iteration in range(1, sk.n_iter_ + 1):
            sk_temp = KMeans(
                n_clusters=k,
                random_state=42,
                max_iter=iteration,
                n_init=1,
            )
            sk_temp.fit(data)
            sk_sse.append(sk_temp.inertia_)

        axes[i].plot(
            range(1, len(alt.sse_history) + 1),
            alt.sse_history,
            label="Alternate",
            marker="o",
        )
        axes[i].plot(
            range(1, len(sk_sse) + 1),
            sk_sse,
            label="Sklearn",
            marker="s",
        )
        axes[i].set_title(f"k={k}")
        axes[i].set_xlabel("Iteration")
        axes[i].set_ylabel("SSE")
        axes[i].legend()

    plt.tight_layout()
    plt.savefig(f"plots/convergence_{dataset_name}.png")
    plt.close()
    print(f"Saved convergence plot for {dataset_name}")


def plot_reassigned(data, dataset_name):
    """
    Plot number of points reassigned per iteration for KMeansAlternate.

    Saves to plots/reassigned_<dataset_name>.png.

    Args:
        data: Array of shape (n_samples, n_features).
        dataset_name: Label for title and filename.
    """
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    fig.suptitle(f"Points Reassigned Per Iteration — {dataset_name}")

    for i, k in enumerate(K_VALUES):
        alt = KMeansAlternate(k=k)
        alt.fit(data)

        axes[i].plot(
            range(1, len(alt.reassigned_history) + 1),
            alt.reassigned_history,
            label="Alternate",
            marker="o",
            color="orange",
        )
        axes[i].set_title(f"k={k}")
        axes[i].set_xlabel("Iteration")
        axes[i].set_ylabel("Points Reassigned")
        axes[i].legend()

    plt.tight_layout()
    plt.savefig(f"plots/reassigned_{dataset_name}.png")
    plt.close()
    print(f"Saved reassigned plot for {dataset_name}")


def plot_runtime(data, dataset_name):
    """
    Plot total fit time vs. k for KMeansAlternate and Sklearn (one run each per k).

    Saves to plots/runtime_<dataset_name>.png.

    Args:
        data: Array of shape (n_samples, n_features).
        dataset_name: Label for title and filename.
    """
    alt_times = []
    sk_times = []

    for k in K_VALUES:
        start = time.time()
        KMeansAlternate(k=k).fit(data)
        alt_times.append(time.time() - start)

        start = time.time()
        KMeans(n_clusters=k, random_state=42, n_init=1).fit(data)
        sk_times.append(time.time() - start)

    plt.figure(figsize=(8, 5))
    plt.plot(K_VALUES, alt_times, label="Alternate", marker="o")
    plt.plot(K_VALUES, sk_times, label="Sklearn", marker="s")
    plt.title(f"Runtime vs K — {dataset_name}")
    plt.xlabel("k")
    plt.ylabel("Time (seconds)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"plots/runtime_{dataset_name}.png")
    plt.close()
    print(f"Saved runtime plot for {dataset_name}")
