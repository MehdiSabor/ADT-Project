import numpy as np

class KMeansAlternate:

    def __init__(self, k, max_iterations=100):
        self.k = k
        self.max_iterations = max_iterations
        self.centroids = None
        self.labels = None
        self.sse_history = []
        self.reassigned_history = []
        self.n_iterations = 0

    #Euclidean Distance 
    def compute_distance(self, point, centroid):
        return np.sqrt(np.sum((point - centroid) ** 2))

    # Squared Distances To All Centroids
    def compute_squared_distances(self, data):
        diffs = data[:, np.newaxis, :] - self.centroids[np.newaxis, :, :]
        return np.sum(diffs ** 2, axis=2)

    # Assign ALL points once before the loop
    def initial_assignment(self, data):
        squared_distances = self.compute_squared_distances(data)
        return np.argmin(squared_distances, axis=1)

    # Update centroids
    def update_centroids(self, data):
        new_centroids = self.centroids.copy()
        counts = np.bincount(self.labels, minlength=self.k)
        non_empty = counts > 0

        sums = np.zeros_like(self.centroids)
        np.add.at(sums, self.labels, data)
        new_centroids[non_empty] = sums[non_empty] / counts[non_empty, np.newaxis]

        return new_centroids

    #  Find furthest point in each cluster 
    def find_furthest_point(self, data, cluster_index):
        cluster_points_idx = np.where(self.labels == cluster_index)[0]
        if len(cluster_points_idx) == 0:
            return None

        cluster_points = data[cluster_points_idx]
        diffs = cluster_points - self.centroids[cluster_index]
        squared_distances = np.sum(diffs ** 2, axis=1)
        return int(cluster_points_idx[np.argmax(squared_distances)])

    # Compute SSE
    def compute_sse(self, data):
        assigned_centroids = self.centroids[self.labels]
        diffs = data - assigned_centroids
        return float(np.sum(diffs ** 2))

    #  Main Fit Function 
    def fit(self, data):
        data = np.asarray(data, dtype=float)

        # Random initialization
        random_indices = np.random.choice(len(data), self.k, replace=False)
        self.centroids = data[random_indices]

        # Initial assignment — done once before the loop
        self.labels = self.initial_assignment(data)

        for iteration in range(self.max_iterations):

            # Step 1 — Update centroids
            self.centroids = self.update_centroids(data)

            # Step 2 — Find and check furthest point per cluster
            reassigned = 0
            for cluster_idx in range(self.k):
                furthest_idx = self.find_furthest_point(data, cluster_idx)
                if furthest_idx is None:
                    continue

                # Check if furthest point is closer to another centroid
                point = data[furthest_idx]
                diffs = self.centroids - point
                squared_distances = np.sum(diffs ** 2, axis=1)
                best_cluster = int(np.argmin(squared_distances))

                if best_cluster != cluster_idx:
                    self.labels[furthest_idx] = best_cluster
                    reassigned += 1

            # Track metrics
            self.sse_history.append(self.compute_sse(data))
            self.reassigned_history.append(reassigned)
            self.n_iterations += 1

            # Convergence check — nothing moved
            if reassigned == 0:
                break

        return self
