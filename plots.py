import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import time
from kmeans_alternate import KMeansAlternate

K_VALUES = [2, 5, 10, 20]

# Reduce to 2D for visualization 
def reduce_to_2d(data):
    pca = PCA(n_components=2)
    return pca.fit_transform(data)

# Plot 1: Cluster Visualization 
def plot_clusters(data, dataset_name):
    data_2d = reduce_to_2d(data)
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    fig.suptitle(f'Cluster Visualization — {dataset_name}')

    for i, k in enumerate(K_VALUES):
        model = KMeansAlternate(k=k)
        model.fit(data)

        axes[i].scatter(data_2d[:, 0], data_2d[:, 1],
                       c=model.labels, cmap='tab10', s=20)
        axes[i].set_title(f'k={k}')
        axes[i].set_xlabel('PC1')
        axes[i].set_ylabel('PC2')

    plt.tight_layout()
    plt.savefig(f'plots/clusters_{dataset_name}.png')
    plt.close()
    print(f"Saved cluster plot for {dataset_name}")

#  Plot 2: Convergence Curve 
def plot_convergence(data, dataset_name):
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    fig.suptitle(f'Convergence Behavior — {dataset_name}')

    for i, k in enumerate(K_VALUES):
        # Alternate variant
        alt = KMeansAlternate(k=k)
        alt.fit(data)

        # Sklearn k-means
        sk = KMeans(n_clusters=k, random_state=42, n_init=1)
        sk.fit(data)
        sk_sse = []
        for iteration in range(1, sk.n_iter_ + 1):
            sk_temp = KMeans(n_clusters=k, random_state=42,
                            max_iter=iteration, n_init=1)
            sk_temp.fit(data)
            sk_sse.append(sk_temp.inertia_)

        axes[i].plot(range(1, len(alt.sse_history) + 1),
                    alt.sse_history, label='Alternate', marker='o')
        axes[i].plot(range(1, len(sk_sse) + 1),
                    sk_sse, label='Sklearn', marker='s')
        axes[i].set_title(f'k={k}')
        axes[i].set_xlabel('Iteration')
        axes[i].set_ylabel('SSE')
        axes[i].legend()

    plt.tight_layout()
    plt.savefig(f'plots/convergence_{dataset_name}.png')
    plt.close()
    print(f"Saved convergence plot for {dataset_name}")

# Plot 3: Points Reassigned Per Iteration 
def plot_reassigned(data, dataset_name):
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    fig.suptitle(f'Points Reassigned Per Iteration — {dataset_name}')

    for i, k in enumerate(K_VALUES):
        alt = KMeansAlternate(k=k)
        alt.fit(data)

        axes[i].plot(range(1, len(alt.reassigned_history) + 1),
                    alt.reassigned_history, 
                    label='Alternate', marker='o', color='orange')
        axes[i].set_title(f'k={k}')
        axes[i].set_xlabel('Iteration')
        axes[i].set_ylabel('Points Reassigned')
        axes[i].legend()

    plt.tight_layout()
    plt.savefig(f'plots/reassigned_{dataset_name}.png')
    plt.close()
    print(f"Saved reassigned plot for {dataset_name}")

# Plot 4: Runtime vs K 
def plot_runtime(data, dataset_name):
    alt_times = []
    sk_times  = []

    for k in K_VALUES:
        # Alternate variant runtime
        start = time.time()
        KMeansAlternate(k=k).fit(data)
        alt_times.append(time.time() - start)

        # Sklearn runtime
        start = time.time()
        KMeans(n_clusters=k, random_state=42, n_init=1).fit(data)
        sk_times.append(time.time() - start)

    plt.figure(figsize=(8, 5))
    plt.plot(K_VALUES, alt_times, label='Alternate', marker='o')
    plt.plot(K_VALUES, sk_times,  label='Sklearn',   marker='s')
    plt.title(f'Runtime vs K — {dataset_name}')
    plt.xlabel('k')
    plt.ylabel('Time (seconds)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'plots/runtime_{dataset_name}.png')
    plt.close()
    print(f"Saved runtime plot for {dataset_name}")