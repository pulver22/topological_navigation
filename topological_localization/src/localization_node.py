

class TopologicalLocalization():

    def __init__(self, unconnected_transition_distance=0.0, sigma_p=0.0, lambda_p=-1, predict=False):
        # num of particles
        self.n_of_ptcl = 300

        # distance of neighboors nodes not connected to allow transitions to
        self.unconnected_transition_distance = unconnected_transition_distance

        # initialization spread of particles
        self.sigma_p = sigma_p

        # parameter for probability of transitioning between nodes
        # if -1 it's set dynamically with picker speed
        self.lambda_p = lambda_p

        # whether to perform prediction only steps when the readings do not change
        self.predict = predict

        # num samples to use to estimate picker speed
        self.speed_samples = 5

        # speed decay when doing only prediction (it does eventually stop)
        self.prediction_speed_decay = 0.95

        self.trajectory_pubs = {}
        self.prediction_pubs = {}
        self.particles_pubs = {}
        self.trajectory = {}
        self.state = {}
        self.node_names = []
        self.node_coords = []
        self.node_diffs2D = []
        self.node_distances = []
        self.connected_nodes = []
        self.P = None
        self.topo_map = None

        rospy.Subscriber("topological_map", TopologicalMap, self.topo_map_cb)

        rospy.loginfo("Waiting for topological map...")
        while self.topo_map is None:
            rospy.sleep(0.5)

        rospy.loginfo("DONE")

        rospy.Subscriber("gps_positions", PeopleStamped, self.gps_cb)


    def topo_map_cb(self, msg):
        """This function receives the Topological Map"""
        self.topo_map = msg
        self.node_names = np.array([node.name for node in self.topo_map.nodes])
        self.node_coords = np.array(
            [[node.pose.position.x, node.pose.position.y] for node in self.topo_map.nodes])
        self.P = self.get_transition_prob_matrix()

    def find_closest_node(self, x, y):
        # print(self.node_coords[:,0].shape)
        closest_node = np.argmin(np.sqrt((self.node_coords[:, 0] - x)**2 + (
            self.node_coords[:, 1] - y)**2))  # closest node to gps location
        closets_node_name = self.node_names[closest_node]

        return closest_node, closets_node_name

    # generate a transition probability function from the topology that can be grounded with values later on
    def get_transition_prob_matrix(self):

        def make_compute_p(node, diffs, distances, connected_nodes, unconnected_neighboors_nodes, lambda_p):
            def _compute_p(speed, tau, only_connected):
                prob = []
                nodes = []

                # check if speed and next node have same direction
                same_direction_mask = np.any(
                    speed * diffs[connected_nodes] >= 0, axis=1)
                pos_connected_nodes = connected_nodes[same_direction_mask]
                neg_connected_nodes = connected_nodes[~same_direction_mask]

                # trans probability goes mostly to connected nodes and one len(connected_nodes)th goes to the unconnected neighboors
                div_coef = len(pos_connected_nodes)
                if (not only_connected) and len(unconnected_neighboors_nodes):
                    div_coef += 1

                    # probability of nodes that are not connected but close enough
                    # project speed vector on the edges toward successive nodes
                    diffs_norm = np.dot(diffs[unconnected_neighboors_nodes],
                                        diffs[unconnected_neighboors_nodes].transpose()).diagonal()
                    speed_proj = (np.dot(diffs[unconnected_neighboors_nodes], speed) / diffs_norm).reshape(
                        (-1, 1)) * diffs[unconnected_neighboors_nodes]
                    speed_proj = np.sqrt(
                        np.dot(speed_proj, speed_proj.transpose()).diagonal())
                    if lambda_p <= 0:
                        _lambda_p = - np.log(0.5) * speed_proj / \
                            distances[unconnected_neighboors_nodes]
                    else:
                        _lambda_p = np.ones(speed_proj.shape) * lambda_p
                    p_move = 1. - np.exp(- tau * _lambda_p)

                    prob += list(((p_move / div_coef) /
                                  len(unconnected_neighboors_nodes)).tolist())
                    nodes += list(unconnected_neighboors_nodes)

                # project speed vector on the edges toward successive nodes
                diffs_norm = np.dot(
                    diffs[pos_connected_nodes], diffs[pos_connected_nodes].transpose()).diagonal()
                speed_proj = (np.dot(diffs[pos_connected_nodes], speed) /
                              diffs_norm).reshape((-1, 1)) * diffs[pos_connected_nodes]
                speed_proj = np.sqrt(
                    np.dot(speed_proj, speed_proj.transpose()).diagonal())

                # probability of moving to trans_node
                # compute lambda considering speed and distance between nodes
                # lambda = (0.6931 * speed) / dist, so that p(transitioning) = 0.5 when the position is halfway between current and next node
                # because exp(-lambda * tau) = 0.5 => ln(0.5) = -0.6931 = - lambda * tau, where tau is time in the node
                if lambda_p <= 0:
                    _lambda_p = - np.log(0.5) * speed_proj / \
                        distances[pos_connected_nodes]
                else:
                    _lambda_p = np.ones(speed_proj.shape) * lambda_p
                p_move = 1. - np.exp(- tau * _lambda_p)

                prob += list((p_move / div_coef).tolist())
                nodes += list(pos_connected_nodes)

                prob += [0.] * len(neg_connected_nodes)
                nodes += list(neg_connected_nodes)

                # set probability of remaining
                prob.append(1.0 - sum(prob))
                nodes.append(node)

                # return probability and list of nodes it can jump to
                return prob, nodes
            return _compute_p

        P = []
        self.node_diffs2D = []
        self.node_distances = []
        self.connected_nodes = []
        # set probability of moving to adjacent node
        for i, _ in enumerate(self.node_names):
            self.node_diffs2D.append(self.node_coords - self.node_coords[i])
            self.node_distances.append(
                np.sqrt(np.sum(self.node_diffs2D[i] ** 2, axis=1)))

            # set probability of moving to adjacent node
            self.connected_nodes.append([np.where(self.node_names == edge.node)[
                                        0][0] for edge in self.topo_map.nodes[i].edges])

            # set probability of unconnected neighboor nodes
            unconnected_neighboors_nodes = []
            if self.unconnected_transition_distance > 0:
                # distances[self.connected_nodes[i] + [i]] = np.inf
                unconnected_neighboors_nodes = np.where(
                    self.node_distances[i] <= self.unconnected_transition_distance)[0].tolist()
                # do not count connected nodes and the current node itself
                for n in self.connected_nodes[i] + [i]:
                    if n in unconnected_neighboors_nodes:
                        unconnected_neighboors_nodes.remove(n)

            P.append(make_compute_p(i, self.node_diffs2D[i][:], self.node_distances[i][:], np.array(
                self.connected_nodes[i]), np.array(unconnected_neighboors_nodes), self.lambda_p))
            # print("node {} ({}), t_list {}, sum all_p_move {}, sum all_p {}".format(i, self.node_names[i], trans_list[i], all_p_move, sum(all_p_move) + p_remain))

        self.node_diffs2D = np.array(self.node_diffs2D)
        self.node_distances = np.array(self.node_distances)
        self.connected_nodes = np.array(self.connected_nodes)

        return P

    def gps_cb(self, msg):
        for person in msg.people:
            if person.person.name not in self.state:
                self.trajectory_pubs[person.person.name] = rospy.Publisher('%s/PF_trajectory' % (
                    person.person.name), MarkerArray, queue_size=10)   # Create a topic for every picker
                self.prediction_pubs[person.person.name] = rospy.Publisher('%s/PF_prediction' % (
                    person.person.name), MarkerArray, queue_size=10)   # Create a topic for every picker
                self.particles_pubs[person.person.name] = rospy.Publisher(
                    '%s/particles' % (person.person.name), MarkerArray, queue_size=10)
                # this here to wait for ROS to register the publishers
                rospy.sleep(0.5)

                closest_node, closest_node_name = self.find_closest_node(
                    person.person.position.x, person.person.position.y)
                rospy.loginfo("Got first gps for {}, closest_node {}".format(
                    person.person.name, closest_node_name))
                #### INITIALIZE PARTICLES ####
                # init new particles object with initial observation
                particles = Particles(
                    self.n_of_ptcl, self.node_coords, sigma_p=self.sigma_p)
                particles.initialize(person.person.position.x, person.person.position.y,
                                     person.header.stamp.to_sec(), sigma_p=self.sigma_p)

                # declare marker arrays
                particles_markerarray = MarkerArray()
                for i in range(self.n_of_ptcl):
                    marker = Marker()
                    marker.header.frame_id = "/map"
                    marker.type = marker.SPHERE
                    marker.pose.position.z = 0
                    marker.pose.orientation.w = 1
                    marker.scale.x = 2
                    marker.scale.y = 2
                    marker.scale.z = 2
                    marker.color.a = 0.6
                    marker.color.r = 1
                    marker.color.g = 0
                    marker.color.b = 0
                    marker.id = i
                    particles_markerarray.markers.append(marker)
                position_markerarray = MarkerArray()
                est_pos = Marker()
                est_pos.header.frame_id = "/map"
                est_pos.type = est_pos.SPHERE
                est_pos.pose.position.z = 6
                est_pos.pose.orientation.w = 1
                est_pos.scale.x = 10
                est_pos.scale.y = 10
                est_pos.scale.z = 10
                est_pos.color.a = 1
                est_pos.color.r = 0
                est_pos.color.b = 1
                est_pos.color.g = 0
                est_pos.id = len(self.state)
                position_markerarray.markers.append(est_pos)
                prediction_markerarray = MarkerArray()
                pred_pos = Marker()
                pred_pos.header.frame_id = "/map"
                pred_pos.type = pred_pos.SPHERE
                pred_pos.pose.position.z = 6
                pred_pos.pose.orientation.w = 1
                pred_pos.scale.x = 10
                pred_pos.scale.y = 10
                pred_pos.scale.z = 10
                pred_pos.color.a = 0.7
                pred_pos.color.r = 0
                pred_pos.color.b = 0
                pred_pos.color.g = 1
                pred_pos.id = len(self.state)
                prediction_markerarray.markers.append(pred_pos)

                self.state.update({
                    person.person.name: {
                        'particles': particles,
                        # history of node estimates
                        'estimated_trajectory': [closest_node]*self.speed_samples,
                        'gps_trajectory': [             # history of gps positions
                            np.array([person.person.position.x,
                                      person.person.position.y])
                        ]*self.speed_samples,
                        'timestamp_trajectory': [       # history of timestamps of gps reading [secs]
                            person.header.stamp.to_sec()
                        ]*self.speed_samples,
                        # current speed of the picker [m/s]
                        'speed': [np.array([0.]*2)]*self.speed_samples,
                        # generate colors for particles of i^th picker
                        'color': np.random.rand(3),
                        'particles_markerarray': particles_markerarray,
                        'position_markerarray': position_markerarray,
                        'prediction_markerarray': prediction_markerarray,
                        "new_reading": True
                    }
                })

                node_estimate = closest_node

                nodes, counts = np.unique(
                    self.state[person.person.name]['particles'].predicted_particles, return_counts=True)
                predicted_pose = np.zeros((2))
                # predicted pose in a covex combination of the particles
                for node, count in zip(nodes, counts):
                    predicted_pose += self.node_coords[node] * \
                        (float(count) / self.n_of_ptcl)

            else:
                ## HERE WE ASSUME THAT IF WE GET THE EXACT SAME MEASUREMENT THEN IT'S NOT A NEW MEASUREMENT
                if person.person.position.x == self.state[person.person.name]['gps_trajectory'][-1][0] and \
                        person.person.position.y == self.state[person.person.name]['gps_trajectory'][-1][1] and \
                        self.predict:
                    self.state[person.person.name]["new_reading"] = False
                else:
                    self.state[person.person.name]["new_reading"] = True

                if self.state[person.person.name]["new_reading"]:
                    # save new received position
                    self.state[person.person.name]['gps_trajectory'].pop(0)
                    self.state[person.person.name]['gps_trajectory'].append(
                        np.array([person.person.position.x, person.person.position.y]))
                    self.state[person.person.name]['timestamp_trajectory'].pop(
                        0)
                    self.state[person.person.name]['timestamp_trajectory'].append(
                        person.header.stamp.to_sec())

                    # compute speed
                    gps_dist = self.state[person.person.name]['gps_trajectory'][-1] - \
                        self.state[person.person.name]['gps_trajectory'][-2]
                    time = self.state[person.person.name]['timestamp_trajectory'][-1] - \
                        self.state[person.person.name]['timestamp_trajectory'][-2]
                    self.state[person.person.name]['speed'].pop(0)
                    if time > 0.:
                        self.state[person.person.name]['speed'].append(
                            gps_dist / time)
                    else:
                        self.state[person.person.name]['speed'].append(
                            self.state[person.person.name]['speed'][-1][:])
                else:
                    self.state[person.person.name]['speed'].pop(0)
                    self.state[person.person.name]['speed'].append(
                        self.state[person.person.name]['speed'][-1] * self.prediction_speed_decay)
                curr_speed = np.average(
                    self.state[person.person.name]['speed'], axis=0)

                rospy.loginfo("Picker: {}, picker speed: {}, new_reading: {}".format(
                    person.person.name, curr_speed, self.state[person.person.name]["new_reading"]))

                #### PREDICTING ####
                if self.P is None:
                    self.P = self.get_transition_prob_matrix()
                # predict new particles position
                if self.state[person.person.name]["new_reading"]:
                    self.state[person.person.name]['particles'].predict(
                        self.P, person.header.stamp.to_sec(), speed=curr_speed, only_connected=False)
                else:
                    self.state[person.person.name]['particles'].predict(
                        self.P, person.header.stamp.to_sec(), speed=curr_speed, only_connected=True)

                # save the number of particles per node and the sum of particles probabilities for each

                #### WEIGHTING ####
                if self.state[person.person.name]["new_reading"]:
                    self.state[person.person.name]['particles'].weight(
                        person.person.position.x, person.person.position.y, self.node_distances, self.connected_nodes, self.node_diffs2D)

                #### UPDATING ESTIMATE ####
                if self.state[person.person.name]["new_reading"]:
                    node_estimate = self.state[person.person.name]['particles'].estimate_node(
                        use_weight=True)
                else:
                    node_estimate = self.state[person.person.name]['particles'].estimate_node(
                        use_weight=False)

                if node_estimate != self.state[person.person.name]['estimated_trajectory'][-1]:
                    self.state[person.person.name]['estimated_trajectory'].pop(
                        0)
                    self.state[person.person.name]['estimated_trajectory'].append(
                        node_estimate)

                #### RESAMPLING #####
                if self.state[person.person.name]["new_reading"]:
                    self.state[person.person.name]['particles'].resample(
                        use_weight=True)
                else:
                    self.state[person.person.name]['particles'].resample(
                        use_weight=False)

                # print("-- resample")
                # print("node estimate: ", node_estimate)
                # nodes, counts = np.unique(self.state[person.person.name]['particles'].particles, return_counts=True)
                # print(self.node_names[nodes], counts)

                # print("-- prediction")
                nodes, counts = np.unique(
                    self.state[person.person.name]['particles'].particles, return_counts=True)
                predicted_pose = np.zeros((2))
                # predicted pose in a covex combination of the particles
                for node, count in zip(nodes, counts):
                    predicted_pose += self.node_coords[node] * \
                        (float(count) / self.n_of_ptcl)

                print("Predicted node {}".format(node_estimate))

            ########### Particle Marker#######
            # Updated particles
            for i, p in enumerate(self.state[person.person.name]['particles'].particles):
                self.state[person.person.name]["particles_markerarray"].markers[i].header.stamp = rospy.get_rostime()
                self.state[person.person.name]["particles_markerarray"].markers[i].pose.position.x = self.node_coords[p][0] + \
                    self.state[person.person.name]["particles_markerarray"].markers[i].scale.x * \
                    np.random.randn(1, 1)  # Updated particles
                self.state[person.person.name]["particles_markerarray"].markers[i].pose.position.y = self.node_coords[p][1] + \
                    self.state[person.person.name]["particles_markerarray"].markers[i].scale.y * \
                    np.random.randn(1, 1)  # Updated particles

            # publish particles
            self.particles_pubs[person.person.name].publish(
                self.state[person.person.name]["particles_markerarray"])

            ################EstMarker###################
            self.state[person.person.name]["position_markerarray"].markers[-1].header.stamp = rospy.get_rostime()
            self.state[person.person.name]["position_markerarray"].markers[-1].pose.position.x = self.node_coords[node_estimate][0]
            self.state[person.person.name]["position_markerarray"].markers[-1].pose.position.y = self.node_coords[node_estimate][1]

            # publish trajectory of positions
            self.trajectory_pubs[person.person.name].publish(
                self.state[person.person.name]["position_markerarray"])

            ################Pred Marker###################
            self.state[person.person.name]["prediction_markerarray"].markers[-1].header.stamp = rospy.get_rostime()
            self.state[person.person.name]["prediction_markerarray"].markers[-1].pose.position.x = predicted_pose[0]
            self.state[person.person.name]["prediction_markerarray"].markers[-1].pose.position.y = predicted_pose[1]

            # publish trajectory of positions
            self.prediction_pubs[person.person.name].publish(
                self.state[person.person.name]["prediction_markerarray"])

            ###~~~~~~~~~~~
            ### log some data (for the paper remove after)
            pickle.dump(
                (rospy.get_rostime(), msg.header.stamp, self.state[person.person.name]["new_reading"], node_estimate, self.node_coords[node_estimate],
                 predicted_pose, self.state[person.person.name]["particles"].predicted_particles, self.state[person.person.name]["particles"].particles),
                self.file_to_save
            )
            ###~~~~~~~~~~~

    def close(self):
        ###~~~~~~~~~~~
        self.file_to_save.close()
        rospy.loginfo("File closed")
        ###~~~~~~~~~~~


if __name__ == "__main__":
    rospy.init_node("topological_localization")

    node = TopologicalLocalization()

    rospy.spin()
    
    node.close()
