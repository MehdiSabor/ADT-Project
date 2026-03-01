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

    # Assign ALL points once before the loop
    def initial_assignment(self, data):
        labels = []
        for point in data:
            distances = [self.compute_distance(point, c) 
                        for c in self.centroids]
            labels.append(np.argmin(distances))
        return np.array(labels)

    # Update centroids
    def update_centroids(self, data):
        new_centroids = []
        for i in range(self.k):
            cluster_points = data[self.labels == i]
            if len(cluster_points) == 0:
                # If cluster is empty keep old centroid
                new_centroids.append(self.centroids[i])
            else:
                new_centroids.append(cluster_points.mean(axis=0))
        return np.array(new_centroids)

    #  Find furthest point in each cluster 
    def find_furthest_point(self, data, cluster_index):
        cluster_points_idx = np.where(self.labels == cluster_index)[0]
        max_dist = -1
        furthest_idx = None
        for idx in cluster_points_idx:
            dist = self.compute_distance(
                data[idx], self.centroids[cluster_index]
            )
            if dist > max_dist:
                max_dist = dist
                furthest_idx = idx
        return furthest_idx

    # Compute SSE
    def compute_sse(self, data):
        sse = 0
        for i in range(self.k):
            cluster_points = data[self.labels == i]
            for point in cluster_points:
                sse += self.compute_distance(point, self.centroids[i]) ** 2
        return sse

    #  Main Fit Function 
    def fit(self, data):
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
                distances = [self.compute_distance(point, c) 
                            for c in self.centroids]
                best_cluster = np.argmin(distances)

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
