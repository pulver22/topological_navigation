import numpy as np

class TopologicalParticleFilter():
    def __init__(self, num, node_coords, sigma_p=0.0):
        self.n_of_ptcl = num
        self.node_coords = node_coords

        # current particles
        self.particles = np.empty((self.n_of_ptcl))
        # previous timestep particles
        self.prev_particles = np.empty((self.n_of_ptcl))
        # particles after prediction phase
        self.predicted_particles = np.empty((self.n_of_ptcl))
        # particles weight
        self.W = np.ones((self.n_of_ptcl))

        # sigma_p is radius spread inital particles
        # sigma_p <= 0     : assigned to closest node
        # sigma_p > 0      : normally distributed between nodes in the radius sigma_p
        # sigma_p = np.inf : equally distributed along all nodes
        self.sigma_p = sigma_p  # [m]

        # time of last update
        self.time = [None] * self.n_of_ptcl
        # life time in current node
        self.life = np.zeros((self.n_of_ptcl))
        # last estimated node
        self.last_estimate = None

    # initialize particles in the map
    def initialize(self, obs_x, obs_y, timestamp_secs, sigma_p=None):
        if sigma_p is None:
            sigma_p = self.sigma_p

        rospy.loginfo("Initialise particles with sigma: {}".format(sigma_p))

        closest_node = np.argmin(np.sqrt(
            (self.node_coords[:, 0] - obs_x)**2 + (self.node_coords[:, 1] - obs_y)**2))
        if sigma_p == np.inf:
            rospy.loginfo(
                "Spreading particles uniformly around nodes. This may be very computationally intensive if the map has many nodes.")
            self.particles = np.random.choice(
                np.arange(self.node_coords.shape[0]), self.n_of_ptcl)
        elif sigma_p <= 0:
            rospy.loginfo("Setting all particles to closest node.")
            self.particles = np.array([closest_node] * self.n_of_ptcl)
        else:
            rospy.loginfo(
                "Spreading particles with a normal distribution in nodes around the initial postion.")
            coef = - np.sum(
                    (self.node_coords - np.array([obs_x, obs_y]))**2, axis=1) / (2 * sigma_p**2)
            nodes_prob = np.exp(coef) / (np.sqrt(2 * np.pi * sigma_p**2))
            nodes_prob /= np.sum(nodes_prob)  # to probability function
            self.particles = np.random.choice(
                np.arange(self.node_coords.shape[0]), self.n_of_ptcl, p=nodes_prob)

        self.prev_particles = self.particles[:]
        self.predicted_particles = self.particles[:]
        self.time = np.ones((self.n_of_ptcl)) * timestamp_secs
        self.life = np.zeros((self.n_of_ptcl))
        self.last_estimate = closest_node

    def predict(self, P, timestamp_secs, speed=0.0, only_connected=False):

        self.life += timestamp_secs - self.time  # update life time
        self.time[:] = timestamp_secs

        for particle_idx in range(self.n_of_ptcl):

            p_node = self.particles[particle_idx]

            transition_p, t_nodes = P[p_node](
                speed=speed, tau=self.life[particle_idx], only_connected=only_connected)

            self.predicted_particles[particle_idx] = np.random.choice(
                t_nodes, p=transition_p)

    # weighting take into account the distance between topological nodes to adapt to different topologies
    def weight(self, obs_x, obs_y, node_distances, connected_nodes, node_diffs2D):
        idx_sort = np.argsort(self.predicted_particles)
        nodes, indices_start, _ = np.unique(
            self.predicted_particles[idx_sort], return_index=True, return_counts=True)
        indices_groups = np.split(idx_sort, indices_start[1:])

        gps_diffs2D = np.abs(self.node_coords - np.array([obs_x, obs_y]))
        gps_distances = np.sqrt(np.sum(gps_diffs2D ** 2, axis=1))

        closest_node = nodes[np.argmin(gps_distances[nodes])]
        # find the edges on which the obs can lie
        candidates = []
        for connected_node in connected_nodes[closest_node]:
            if node_diffs2D[closest_node][connected_node][0] == (gps_diffs2D[closest_node][0] + gps_diffs2D[connected_node][0]) or\
                    node_diffs2D[closest_node][connected_node][1] == (gps_diffs2D[closest_node][1] + gps_diffs2D[connected_node][1]):
                candidates.append(connected_node)
        if len(candidates) > 0:
            distance_reference = np.average(
                node_distances[closest_node][candidates])
        else:
            distance_reference = np.average(
                node_distances[closest_node][connected_nodes[closest_node]])

        gamma = np.log(0.5) / (distance_reference / 2)
        D = np.zeros((self.n_of_ptcl))
        for _, (node, indices) in enumerate(zip(nodes, indices_groups)):
            D[indices] = gamma * gps_distances[node]
        D = np.exp(D)

        # D = np.zeros((self.n_of_ptcl))
        # for _, (node, indices) in enumerate(zip(nodes, indices_groups)):
        #     D[indices] = max(0., 1 - gps_distances[node] / distance_reference)

        self.W = normalize(D)               # Covert distance to weights

    # produce the node estimate based on topological mass from particles and their weight
    def estimate_node(self, use_weight=True):
        nodes, indices_start, counts = np.unique(
            self.predicted_particles, return_index=True, return_counts=True)
        masses = []
        if use_weight:
            for (_, index_start, count) in zip(nodes, indices_start, counts):
                masses.append(self.W[index_start] * count)
        else:
            masses = counts
        self.last_estimate = nodes[np.argmax(masses)]

        return self.last_estimate

    def resample(self, use_weight=True):
        # pass
        self.prev_particles = self.particles[:]
        if use_weight:
            self.particles = np.random.choice(
                self.predicted_particles, self.n_of_ptcl, p=self.W)
        else:
            self.particles = self.predicted_particles[:]

        ## reset life time if particle jumped in another node
        for particle_idx in range(self.n_of_ptcl):
            if self.particles[particle_idx] != self.prev_particles[particle_idx]:
                self.life[particle_idx] = 0
