import numpy as np

class KMeansCustom:

    def __init__(self, k, max_iterations=100, epsilon=1e-4):
        self.k = k
        self.max_iterations = max_iterations
        self.epsilon = epsilon        # drift threshold
        self.centroids = None
        self.labels = None
        self.sse_history = []
        self.reassigned_history = []
        self.n_iterations = 0

    # ─── Euclidean Distance ───────────────────────────────
    def compute_distance(self, point, centroid):
        return np.sqrt(np.sum((point - centroid) ** 2))

    # ─── ESPI Initialization ──────────────────────────────
    def espi_init(self, data):
        # Step 1: find dominant dimension (largest variance)
        variances = np.var(data, axis=0)
        dominant_dim = np.argmax(variances)

        # Step 2: sort points along dominant dimension
        sorted_indices = np.argsort(data[:, dominant_dim])
        sorted_data = data[sorted_indices]

        # Step 3: divide into k equal partitions
        partitions = np.array_split(sorted_data, self.k)

        # Step 4: compute mean of each partition as centroid
        centroids = []
        for partition in partitions:
            if len(partition) == 0:
                # if empty partition pick random point
                centroids.append(data[np.random.randint(len(data))])
            else:
                centroids.append(partition.mean(axis=0))

        return np.array(centroids)

    # ─── Initial Full Assignment ──────────────────────────
    def full_assignment(self, data):
        labels = []
        for point in data:
            distances = [self.compute_distance(point, c)
                        for c in self.centroids]
            labels.append(np.argmin(distances))
        return np.array(labels)

    # ─── Update Centroids ─────────────────────────────────
    def update_centroids(self, data):
        new_centroids = []
        for i in range(self.k):
            cluster_points = data[self.labels == i]
            if len(cluster_points) == 0:
                new_centroids.append(self.centroids[i])
            else:
                new_centroids.append(cluster_points.mean(axis=0))
        return np.array(new_centroids)

    # ─── Compute Drift Per Centroid ───────────────────────
    def compute_drift(self, old_centroids, new_centroids):
        # how much each centroid moved this iteration
        drift = []
        for i in range(self.k):
            d = self.compute_distance(old_centroids[i], 
                                      new_centroids[i])
            drift.append(d)
        return np.array(drift)

    # ─── Find Points To Check Using Drift ─────────────────
    def get_active_points(self, data, drift):
        # a point is active (needs checking) if:
        # its distance to centroid <= 2 * drift of that centroid
        # meaning it MIGHT switch clusters
        active = []
        for idx, point in enumerate(data):
            cluster = self.labels[idx]
            dist_to_centroid = self.compute_distance(
                point, self.centroids[cluster]
            )
            # if point is close enough to boundary → check it
            if dist_to_centroid <= 2 * drift[cluster] + self.epsilon:
                active.append(idx)
        return active

    # ─── Compute SSE ──────────────────────────────────────
    def compute_sse(self, data):
        sse = 0
        for i in range(self.k):
            cluster_points = data[self.labels == i]
            for point in cluster_points:
                sse += self.compute_distance(
                    point, self.centroids[i]) ** 2
        return sse

    # ─── Main Fit ─────────────────────────────────────────
    def fit(self, data):

        # ── Phase 1: ESPI Initialization ──────────────────
        self.centroids = self.espi_init(data)

        # ── Phase 2: Initial Full Assignment ──────────────
        self.labels = self.full_assignment(data)

        # ── Phase 3: Iterative Loop ────────────────────────
        for iteration in range(self.max_iterations):

            old_centroids = self.centroids.copy()

            # Step 1: update centroids
            self.centroids = self.update_centroids(data)

            # Step 2: compute how much each centroid moved
            drift = self.compute_drift(old_centroids, 
                                       self.centroids)

            # Step 3: find only points that might switch
            # skip points deep inside their cluster
            active_indices = self.get_active_points(data, drift)

            # Step 4: reassign only active points
            reassigned = 0
            for idx in active_indices:
                point = data[idx]
                distances = [self.compute_distance(point, c)
                            for c in self.centroids]
                new_label = np.argmin(distances)
                if new_label != self.labels[idx]:
                    self.labels[idx] = new_label
                    reassigned += 1

            # track metrics
            self.sse_history.append(self.compute_sse(data))
            self.reassigned_history.append(reassigned)
            self.n_iterations += 1

            # convergence: nothing moved or centroids stable
            max_drift = np.max(drift)
            if reassigned == 0 or max_drift < self.epsilon:
                break

        return self
