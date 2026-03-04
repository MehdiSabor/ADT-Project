"""
Alternate K-Means variant: furthest-point reassignment per cluster.

This module implements a K-Means variant that differs from the standard
algorithm in its iteration strategy:
  - Initialization: random selection of k data points as initial centroids.
  - Each iteration: update centroids from current assignments, then for each
    cluster find the single point furthest from its centroid; if that point
    is closer to another centroid, reassign it. Only these "furthest" points
    are considered for reassignment each round.

This conservative strategy typically converges in more iterations than
full reassignment and is used for comparison in the project (convergence
curves, reassignment counts, runtime).
"""

import numpy as np


class KMeansAlternate:
    """
    K-Means variant that reassigns only the furthest point per cluster each iteration.

    Attributes:
        k (int): Number of clusters.
        max_iterations (int): Maximum number of iterations.
        centroids (np.ndarray | None): Final cluster centers, shape (k, n_features).
        labels (np.ndarray | None): Cluster index per point, shape (n_samples,).
        sse_history (list): SSE after each iteration.
        reassigned_history (list): Number of points reassigned each iteration.
        n_iterations (int): Number of iterations performed in last fit.
    """

    def __init__(self, k, max_iterations=100):
        """
        Initialize the alternate K-Means model.

        Args:
            k: Number of clusters.
            max_iterations: Maximum number of iterations.
        """
        self.k = k
        self.max_iterations = max_iterations
        self.centroids = None
        self.labels = None
        self.sse_history = []
        self.reassigned_history = []
        self.n_iterations = 0

    def compute_distance(self, point, centroid):
        """
        Compute Euclidean distance between a point and a centroid.

        Args:
            point: 1D array of shape (n_features,).
            centroid: 1D array of shape (n_features,).

        Returns:
            float: Euclidean distance.
        """
        return np.sqrt(np.sum((point - centroid) ** 2))

    def compute_squared_distances(self, data):
        """
        Compute squared distance from each point to each centroid.

        Args:
            data: Array of shape (n_samples, n_features).

        Returns:
            np.ndarray: Shape (n_samples, k), squared distances.
        """
        diffs = data[:, np.newaxis, :] - self.centroids[np.newaxis, :, :]
        return np.sum(diffs ** 2, axis=2)

    def initial_assignment(self, data):
        """
        Assign every point to its nearest centroid (used once after init).

        Args:
            data: Array of shape (n_samples, n_features).

        Returns:
            np.ndarray: Cluster indices, shape (n_samples,).
        """
        squared_distances = self.compute_squared_distances(data)
        return np.argmin(squared_distances, axis=1)

    def update_centroids(self, data):
        """
        Recompute centroids as the mean of points in each cluster.

        Empty clusters are left unchanged.

        Args:
            data: Array of shape (n_samples, n_features).

        Returns:
            np.ndarray: New centroids, shape (k, n_features).
        """
        new_centroids = self.centroids.copy()
        counts = np.bincount(self.labels, minlength=self.k)
        non_empty = counts > 0

        sums = np.zeros_like(self.centroids)
        np.add.at(sums, self.labels, data)
        new_centroids[non_empty] = sums[non_empty] / counts[non_empty, np.newaxis]

        return new_centroids

    def find_furthest_point(self, data, cluster_index):
        """
        Find the index of the point in the given cluster that is furthest
        from the cluster's centroid.

        Args:
            data: Array of shape (n_samples, n_features).
            cluster_index: Which cluster (0 .. k-1).

        Returns:
            int | None: Index of the furthest point, or None if cluster is empty.
        """
        cluster_points_idx = np.where(self.labels == cluster_index)[0]
        if len(cluster_points_idx) == 0:
            return None

        cluster_points = data[cluster_points_idx]
        diffs = cluster_points - self.centroids[cluster_index]
        squared_distances = np.sum(diffs ** 2, axis=1)
        return int(cluster_points_idx[np.argmax(squared_distances)])

    def compute_sse(self, data):
        """
        Compute sum of squared errors (within-cluster squared distances).

        Args:
            data: Array of shape (n_samples, n_features).

        Returns:
            float: Total SSE.
        """
        assigned_centroids = self.centroids[self.labels]
        diffs = data - assigned_centroids
        return float(np.sum(diffs ** 2))

    def fit(self, data):
        """
        Run alternate K-Means: random init, then each iteration update
        centroids and reassign only the furthest point per cluster if
        it is closer to another centroid. Stop when no reassignments occur.

        Args:
            data: Array-like of shape (n_samples, n_features).

        Returns:
            self: Fitted estimator.
        """
        data = np.asarray(data, dtype=float)

        # Random initialization: k distinct data points as centroids
        random_indices = np.random.choice(len(data), self.k, replace=False)
        self.centroids = data[random_indices].copy()

        # One full assignment before the main loop
        self.labels = self.initial_assignment(data)

        for iteration in range(self.max_iterations):
            # Step 1: update centroids from current assignments
            self.centroids = self.update_centroids(data)

            # Step 2: for each cluster, check if furthest point should move
            reassigned = 0
            for cluster_idx in range(self.k):
                furthest_idx = self.find_furthest_point(data, cluster_idx)
                if furthest_idx is None:
                    continue

                # Distance from this point to every centroid
                point = data[furthest_idx]
                diffs = self.centroids - point
                squared_distances = np.sum(diffs ** 2, axis=1)
                best_cluster = int(np.argmin(squared_distances))

                if best_cluster != cluster_idx:
                    self.labels[furthest_idx] = best_cluster
                    reassigned += 1

            self.sse_history.append(self.compute_sse(data))
            self.reassigned_history.append(reassigned)
            self.n_iterations += 1

            if reassigned == 0:
                break

        return self
