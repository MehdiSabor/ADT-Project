"""
Custom K-Means implementation with deterministic percentile-based initialization.

This module implements a variant of K-Means that uses:
  - SPI-style initialization: centroids are placed along the dimension with
    highest variance, spaced between low and high percentiles (e.g., 5th–95th).
  - Full reassignment every iteration (standard assign-then-update loop).
  - Stopping when no points change cluster or maximum centroid drift is below
    a small epsilon.

No random restarts are needed due to deterministic initialization. Suitable
for reproducible clustering and comparison with random-initialization methods.
"""

import numpy as np


class KMeansCustom:
    """
    K-Means with percentile-based (SPI) initialization and drift-based stopping.

    Attributes:
        k (int): Number of clusters.
        max_iterations (int): Maximum number of assign-update iterations.
        epsilon (float): Stop when max centroid drift < epsilon.
        low_percentile (float): Lower percentile for initial centroid spread.
        high_percentile (float): Upper percentile for initial centroid spread.
        centroids (np.ndarray | None): Final cluster centers, shape (k, n_features).
        labels (np.ndarray | None): Cluster index per point, shape (n_samples,).
        sse_history (list): SSE after each iteration.
        drift_history (list): Max centroid drift after each iteration.
        reassigned_history (list): Number of points that changed cluster each iteration.
        n_iterations (int): Number of iterations performed in last fit.
    """

    def __init__(
        self,
        k,
        max_iterations=100,
        epsilon=1e-4,
        low_percentile=5,
        high_percentile=95,
    ):
        """
        Initialize the custom K-Means model.

        Args:
            k: Number of clusters.
            max_iterations: Cap on assign-update iterations.
            epsilon: Convergence threshold for maximum centroid drift.
            low_percentile: Lower bound percentile for initial centroid placement.
            high_percentile: Upper bound percentile for initial centroid placement.
        """
        self.k = k
        self.max_iterations = max_iterations
        self.epsilon = epsilon
        self.low_percentile = low_percentile
        self.high_percentile = high_percentile

        # Set by fit()
        self.centroids = None
        self.labels = None
        self.sse_history = []
        self.drift_history = []
        self.reassigned_history = []
        self.n_iterations = 0

    # ─── Euclidean Distance ───────────────────────────────
    def compute_distance(self, point, centroid):
        """
        Compute Euclidean distance between a single point and a centroid.

        Args:
            point: 1D array of shape (n_features,).
            centroid: 1D array of shape (n_features,).

        Returns:
            float: Euclidean distance.
        """
        return np.sqrt(np.sum((point - centroid) ** 2))

    # ─── Squared Distances To All Centroids ───────────────
    def compute_squared_distances(self, data):
        """
        Compute squared Euclidean distance from each point to each centroid.

        Uses broadcasting: avoids Python loops for speed. Returns squared
        distances so we can assign by argmin without taking sqrt.

        Args:
            data: Array of shape (n_samples, n_features).

        Returns:
            np.ndarray: Shape (n_samples, k), squared distances.
        """
        diffs = data[:, np.newaxis, :] - self.centroids[np.newaxis, :, :]
        return np.sum(diffs ** 2, axis=2)

    # ─── Percentile Initialization Across All Features ───
    def percentile_init(self, data):
        """
        Initialize centroids using the dominant-axis percentile (SPI) strategy.

        Finds the feature dimension with maximum variance, then places k
        centroids at evenly spaced positions between the low and high
        percentiles along that dimension. Other dimensions are set to the
        global mean (so centroids differ only along the dominant axis).

        Args:
            data: Array of shape (n_samples, n_features).

        Returns:
            np.ndarray: Initial centroids, shape (k, n_features).
        """
        variances = np.var(data, axis=0)
        dominant_dim = int(np.argmax(variances))

        base_centroid = np.mean(data, axis=0)
        low = np.percentile(data[:, dominant_dim], self.low_percentile)
        high = np.percentile(data[:, dominant_dim], self.high_percentile)

        # Start all centroids at the global mean, then vary only the dominant dimension
        centroids = np.tile(base_centroid, (self.k, 1))
        for j in range(self.k):
            t = (j + 0.5) / self.k
            centroids[j, dominant_dim] = low + t * (high - low)

        return centroids

    # ─── Full Assignment ──────────────────────────────────
    def full_assignment(self, data):
        """
        Assign every point to its nearest centroid (by squared distance).

        Args:
            data: Array of shape (n_samples, n_features).

        Returns:
            np.ndarray: Integer cluster indices, shape (n_samples,).
        """
        squared_distances = self.compute_squared_distances(data)
        return np.argmin(squared_distances, axis=1)

    # ─── Update Centroids ─────────────────────────────────
    def update_centroids(self, data):
        """
        Recompute each centroid as the mean of points assigned to that cluster.

        Empty clusters are left unchanged (no points to average).

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

    # ─── Compute Drift Per Centroid ───────────────────────
    def compute_drift(self, old_centroids, new_centroids):
        """
        Compute Euclidean norm of each centroid's movement (drift).

        Args:
            old_centroids: Centroids before update, shape (k, n_features).
            new_centroids: Centroids after update, shape (k, n_features).

        Returns:
            np.ndarray: Drift per centroid, shape (k,).
        """
        diffs = old_centroids - new_centroids
        return np.sqrt(np.sum(diffs ** 2, axis=1))

    # ─── Compute SSE ──────────────────────────────────────
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

    # ─── Main Fit ─────────────────────────────────────────
    def fit(self, data):
        """
        Run custom K-Means until convergence or max_iterations.

        Steps: (1) percentile init, (2) initial full assignment,
        (3) initial centroid update, then loop: (4) full assignment,
        (5) centroid update, (6) drift and SSE tracking, (7) stop if
        no reassignments or max drift < epsilon.

        Args:
            data: Array-like of shape (n_samples, n_features).

        Returns:
            self: Fitted estimator (centroids and labels set).
        """
        data = np.asarray(data, dtype=float)

        # Step 1: deterministic percentile-based initialization
        self.centroids = self.percentile_init(data)

        # Step 2: initial full assignment
        self.labels = self.full_assignment(data)

        # Step 3: align centroids with the first assignment
        self.centroids = self.update_centroids(data)

        self.sse_history = []
        self.drift_history = []
        self.reassigned_history = []
        self.n_iterations = 0

        for _ in range(self.max_iterations):
            old_centroids = self.centroids.copy()
            old_labels = self.labels.copy()

            # Step 4: assign every point to nearest centroid
            self.labels = self.full_assignment(data)

            # Count how many labels changed (for diagnostics)
            reassigned = int(np.sum(self.labels != old_labels))

            # Step 5: recompute centroids
            self.centroids = self.update_centroids(data)

            # Step 6: compute centroid drift for stopping criterion
            drift = self.compute_drift(old_centroids, self.centroids)
            max_drift = np.max(drift)

            # Record metrics for plotting/analysis
            self.sse_history.append(self.compute_sse(data))
            self.drift_history.append(max_drift)
            self.reassigned_history.append(reassigned)
            self.n_iterations += 1

            # Step 7: stop if converged (no moves or negligible drift)
            if reassigned == 0 or max_drift < self.epsilon:
                break

        return self

    # ─── Predict ──────────────────────────────────────────
    def predict(self, data):
        """
        Assign new points to the nearest fitted centroid.

        Args:
            data: Array-like of shape (n_samples, n_features).

        Returns:
            np.ndarray: Cluster indices, shape (n_samples,).
        """
        data = np.asarray(data, dtype=float)
        squared_distances = self.compute_squared_distances(data)
        return np.argmin(squared_distances, axis=1)
