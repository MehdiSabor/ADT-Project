import numpy as np


class KMeansCustom:
    def __init__(
        self,
        k,
        max_iterations=100,
        epsilon=1e-4,
        low_percentile=5,
        high_percentile=95
    ):
        self.k = k
        self.max_iterations = max_iterations
        self.epsilon = epsilon

        # Percentile-based initialization
        self.low_percentile = low_percentile
        self.high_percentile = high_percentile

        self.centroids = None
        self.labels = None
        self.sse_history = []
        self.drift_history = []
        self.reassigned_history = []
        self.n_iterations = 0

    # ─── Euclidean Distance ───────────────────────────────
    def compute_distance(self, point, centroid):
        return np.sqrt(np.sum((point - centroid) ** 2))

    # ─── Squared Distances To All Centroids ───────────────
    def compute_squared_distances(self, data):
        diffs = data[:, np.newaxis, :] - self.centroids[np.newaxis, :, :]
        return np.sum(diffs ** 2, axis=2)

    # ─── Percentile Initialization Across All Features ───
    def percentile_init(self, data):
        variances = np.var(data, axis=0)
        dominant_dim = int(np.argmax(variances))

        base_centroid = np.mean(data, axis=0)
        low = np.percentile(data[:, dominant_dim], self.low_percentile)
        high = np.percentile(data[:, dominant_dim], self.high_percentile)

        centroids = np.tile(base_centroid, (self.k, 1))
        for j in range(self.k):
            t = (j + 0.5) / self.k
            centroids[j, dominant_dim] = low + t * (high - low)

        return centroids

    # ─── Full Assignment ──────────────────────────────────
    def full_assignment(self, data):
        squared_distances = self.compute_squared_distances(data)
        return np.argmin(squared_distances, axis=1)

    # ─── Update Centroids ─────────────────────────────────
    def update_centroids(self, data):
        new_centroids = self.centroids.copy()
        counts = np.bincount(self.labels, minlength=self.k)
        non_empty = counts > 0

        sums = np.zeros_like(self.centroids)
        np.add.at(sums, self.labels, data)
        new_centroids[non_empty] = sums[non_empty] / counts[non_empty, np.newaxis]

        return new_centroids

    # ─── Compute Drift Per Centroid ───────────────────────
    def compute_drift(self, old_centroids, new_centroids):
        diffs = old_centroids - new_centroids
        return np.sqrt(np.sum(diffs ** 2, axis=1))

    # ─── Compute SSE ──────────────────────────────────────
    def compute_sse(self, data):
        assigned_centroids = self.centroids[self.labels]
        diffs = data - assigned_centroids
        return float(np.sum(diffs ** 2))

    # ─── Main Fit ─────────────────────────────────────────
    def fit(self, data):
        data = np.asarray(data, dtype=float)

        # Step 1: percentile-based initialization
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

            # Count how many labels changed
            reassigned = int(np.sum(self.labels != old_labels))

            # Step 5: recompute centroids
            self.centroids = self.update_centroids(data)

            # Step 6: compute centroid drift
            drift = self.compute_drift(old_centroids, self.centroids)
            max_drift = np.max(drift)

            # Track metrics
            self.sse_history.append(self.compute_sse(data))
            self.drift_history.append(max_drift)
            self.reassigned_history.append(reassigned)
            self.n_iterations += 1

            # Step 7: stop if centroids barely move
            if reassigned == 0 or max_drift < self.epsilon:
                break

        return self

    # ─── Predict ──────────────────────────────────────────
    def predict(self, data):
        data = np.asarray(data, dtype=float)
        squared_distances = self.compute_squared_distances(data)
        return np.argmin(squared_distances, axis=1)
