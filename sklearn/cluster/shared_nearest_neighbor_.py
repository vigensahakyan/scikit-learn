"""

Shared Nearest Neighbor (SNN) clustering.
Original paper: https://www-users.cs.umn.edu/~kumar001/papers/kdd02_snn_28.pdf

"""

# Authors: Vigen Sahakyan <vigen.sahakyan92@gmail.com>
# License: BSD 3 clause


from ..base import BaseEstimator, ClusterMixin
from ..utils import check_array, check_consistent_length
from ..utils.testing import ignore_warnings
from ..neighbors import NearestNeighbors


def snn_clustering(X, k=5, eps=5, min_pts=10, metric='minkowski', metric_params=None,
                   algorithm='auto', leaf_size=30, p=2, sample_weight=None,
                   n_jobs=None):
    """Perform Shared Nearest Neighbor (SNN) clustering from vector array or distance matrix.

        Read more in the :ref:`User Guide <snn_clustering>`. # TODO create doc in user guide

        Parameters
        ----------
        X : array or sparse (CSR) matrix of shape (n_samples, n_features), or \
                array of shape (n_samples, n_samples)
            A feature array, or array of distances between samples if
            ``metric='precomputed'``.

        k_neighbors : int, optional
            Amount of nearest neighbors to use when sparsifying matrix.

        eps : float, optional
            The maximum distance between two samples for them to be considered
            as in the same neighborhood.

        min_samples : int, optional
            The number of samples (or total weight) in a neighborhood for a point
            to be considered as a core point. This includes the point itself.

        metric : string, or callable
            The metric to use when calculating distance between instances in a
            feature array. If metric is a string or callable, it must be one of
            the options allowed by :func:`sklearn.metrics.pairwise_distances` for
            its metric parameter.
            If metric is "precomputed", X is assumed to be a distance matrix and
            must be square. X may be a sparse matrix, in which case only "nonzero"
            elements may be considered neighbors for DBSCAN.

        metric_params : dict, optional
            Additional keyword arguments for the metric function.

            .. versionadded:: 0.19

        algorithm : {'auto', 'ball_tree', 'kd_tree', 'brute'}, optional
            The algorithm to be used by the NearestNeighbors module
            to compute pointwise distances and find nearest neighbors.
            See NearestNeighbors module documentation for details.

        leaf_size : int, optional (default = 30)
            Leaf size passed to BallTree or cKDTree. This can affect the speed
            of the construction and query, as well as the memory required
            to store the tree. The optimal value depends
            on the nature of the problem.

        p : float, optional
            The power of the Minkowski metric to be used to calculate distance
            between points.

        sample_weight : array, shape (n_samples,), optional
            Weight of each sample, such that a sample with a weight of at least
            ``min_samples`` is by itself a core sample; a sample with negative
            weight may inhibit its eps-neighbor from being core.
            Note that weights are absolute, and default to 1.

        n_jobs : int or None, optional (default=None)
            The number of parallel jobs to run for neighbors search.
            ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
            ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
            for more details.

        Returns
        -------
        core_samples : array [n_core_samples]
            Indices of core samples.

        labels : array [n_samples]
            Cluster labels for each point.  Noisy samples are given the label -1.

        See also
        --------
        DBSCAN
            An estimator interface for this clustering algorithm.
        optics
            A similar clustering at multiple values of eps. Our implementation
            is optimized for memory usage.

        Notes
        -----
        For an example, see :ref:`examples/cluster/plot_dbscan.py
        <sphx_glr_auto_examples_cluster_plot_dbscan.py>`.

        This implementation bulk-computes all neighborhood queries, which increases
        the memory complexity to O(n.d) where d is the average number of neighbors,
        while original DBSCAN had memory complexity O(n). It may attract a higher
        memory complexity when querying these nearest neighborhoods, depending
        on the ``algorithm``.

        One way to avoid the query complexity is to pre-compute sparse
        neighborhoods in chunks using
        :func:`NearestNeighbors.radius_neighbors_graph
        <sklearn.neighbors.NearestNeighbors.radius_neighbors_graph>` with
        ``mode='distance'``, then using ``metric='precomputed'`` here.

        Another way to reduce memory and computation time is to remove
        (near-)duplicate points and use ``sample_weight`` instead.

        :func:`cluster.optics <sklearn.cluster.optics>` provides a similar
        clustering with lower memory usage.

        References
        ----------
        L. Ertoz, M. Steinbach, and V. Kumar. "Finding Clusters of Different Sizes,
        Shapes, and Densities in Noisy, High Dimensional Data".
        In: Proceedings of the Third SIAM International Conference on Data Mining
        (SDM 2003), volume 112 of Proceedings in Applied Mathematics,
        Society for Industrial and Applied Mathematics, (2003)
        """

    if k < 0.0 or k > X.shape[0]:
        raise ValueError("k has to be in range [0, {0}]".format(X.shape[0]))

    if not eps > 0.0:
        raise ValueError("eps must be positive.")

    X = check_array(X, accept_sparse='csr')
    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight)
        check_consistent_length(X, sample_weight)





class SNN(BaseEstimator, ClusterMixin):

    def __init__(self, k, eps, min_pts, metric='minkowski', metric_params=None,
                 algorithm='auto', leaf_size=30, p=2, sample_weight=None,
                 n_jobs=None):
        pass

    def fit(self, X, y=None, sample_weight=None):
        pass

    def fit_predict(self, X, y=None, sample_weight=None):
        pass


from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import time
from queue import Queue
import time

import collections

"""
1. Compute the similarity matrix. (This corresponds to a
similarity graph with data points for nodes and edges whose
weights are the similarities between data points.)
2. Sparsify the similarity matrix by keeping only the k most
similar neighbors. (This corresponds to only keeping the k
strongest links of the similarity graph.)
3. Construct the shared nearest neighbor graph from the
sparsified similarity matrix. (Each link is assigned a strength
according to the formula in Section 3.)
4. For every node (data point) in the graph, calculate the total
strength of links coming out of the point. (Steps 1-4 are
identical to the Jarvis – Patrick scheme.)
5. Identify representative points by choosing the points that
have high density, i.e., high total link strength.
6. Identify noise points by choosing the points that have low
density (total link strength) and remove them.
7. Remove all links between points that have weight smaller
than a threshold.
8. Take connected components of points to form clusters,
where every point in a cluster is either a representative point
or is connected to a representative point

"""


class FuncTimer(object):
    def __init__(self, txt_message):
        self.txt_message = txt_message

    def __enter__(self):
        self.start = time.time()

    def __exit__(self, exception_type, exception_value, traceback):
        self.end = time.time()
        result_time = (self.end - self.start)
        print('{0} {1}'.format(self.txt_message, result_time))
        if exception_value is not None:
            raise exception_value
        return result_time


class SNNClustering(object):

    def __init__(self, k, eps, min_pts):
        self.X = None
        self.k = k
        self.eps = eps
        self.min_pts = min_pts
        if self.min_pts > self.k:
            raise BaseException("HEY YOU ->>. Look at my eyes, "
                                "If you put min_pts more than K that's mean "
                                "you don't understand algorithm !!! Go and "
                                "read that shithole paper again !")
        self.N = self.X.shape[0]

        self.dist_matrix = None
        self.nn_dict = None
        self.snn_graph_matrix = None
        self.densities = None
        self.core_points = None
        self.labels = None

    def create_distance_matrix(self):
        # TODO with NearestNeighbors
        self.dist_matrix = 1 - cosine_similarity(self.X, self.X)

    def sparsify_distance_matrix(self):
        # TODO Merge with above (create_distance_matrix) by using
        # NearestNeighbor models
        if self.dist_matrix is None:
            raise BaseException('Distance Matrix is None first you '
                                'need to run self.create_distance_matrix()')

        self.nn_dict = {}

        for i, dist_vector in enumerate(self.dist_matrix):
            top_ids = np.argsort(dist_vector)[:self.k]
            self.nn_dict[i] = set(top_ids.tolist())

            # print("start ")
            # print(titles[i])
            # scores_sorted = np.sort(dist_vector)
            # top_titles = titles[top_ids]
            # top_txt_lens = text_lens[top_ids]
            # # print(titles[test_id])
            # for idx, (a_len, t, score) in enumerate(
            #         zip(top_txt_lens, top_titles, scores_sorted)):
            #     print('{0} | {1} | {2}'.format(a_len, score, t))
            #     # if idx == 15:
            #     #     break
            # print("###############")

    def construct_snn_graph(self):
        if self.nn_dict is None:
            raise BaseException('Nearest Neighbor dict is none you need to'
                                'run self.sparsify_distance_matrix() first.')
        print("xx, :", self.N)
        self.snn_graph_matrix = [[0 for i in range(self.N)]
                                 for j in range(self.N)]
        print("size {}".format(self.N))
        print(len(self.snn_graph_matrix))
        points = set()
        print("CORE COUNT: ", points)

        for i in range(self.N):
            for j in range(self.N):
                i_set = self.nn_dict[i]
                j_set = self.nn_dict[j]
                intrsct_cnt = len(i_set.intersection(j_set))

                if (j in i_set) & (i in j_set):
                    points.add(i)
                    self.snn_graph_matrix[i][j] = intrsct_cnt

        print("CORE COUNT: ", len(points))

    def find_snn_densities(self):
        if self.snn_graph_matrix is None:
            raise BaseException('SNN graph is none you need to'
                                'run self.construct_snn_graph() first.')
        self.densities = [0 for i in range(self.N)]

        for idx, row in enumerate(self.snn_graph_matrix):
            density = 0
            for v in row:
                if v > 0:
                    density += 1
            self.densities[idx] = density

        # for id, density in enumerate(self.densities):
        #     print(id, density)

    def find_core_points(self):
        if self.densities is None or self.densities == []:
            raise BaseException('Densities list is None or empty check your '
                          'parameters and make sure you are run'
                          'find_snn_densities() before running this function')

        print("durum")
        self.core_points = [False for i in range(self.N)]
        print("durum 1")

        cnt = 0
        for idx, v in enumerate(self.densities):
            if v >= self.min_pts:
                self.core_points[idx]=True
                cnt+=1

        print('Number of core points is : {}'.format(cnt))

    def form_clusters_from_core_points(self):
        C = 0
        visited_cores = set()
        self.labels = [-1 for i in range(self.N)]
        for idx, v in enumerate(self.core_points):
            if v is True:
                if idx in visited_cores:
                    continue
                else:
                    visited_cores.add(idx)
                    C += 1
                    self.labels[idx] = C

                    neighbors = self.find_core_neighbors(core_id=idx)
                    self.expand_clusters(neighbors, visited_cores, C)

    def remove_noise_assign_core_points(self):
        res_cnt = collections.Counter(self.labels)
        print("KKKKKK", len(res_cnt))
        for i in range(self.N):
            not_noise = False
            max_sim = -133333
            best_core_id = -2
            sim = None
            print("XXXX : ",i)
            if self.core_points[i] is True:
                continue

            for idx, v in enumerate(self.core_points):

                if v is True:
                    p = idx
                    sim = self.snn_graph_matrix[i][p]
                    if sim >= self.eps:
                        not_noise = True
                    if sim > max_sim:
                        max_sim = sim
                        best_core_id = p

            if not_noise:
                self.labels[i] = self.labels[best_core_id]
                print("TTTGT: ", best_core_id)

        res_cnt = collections.Counter(self.labels)
        print(len(res_cnt))

    def find_core_neighbors(self, core_id):
        neighbors = []
        # neighbors = Queue()

        for idx, v in enumerate(self.core_points):
            if v is True:
                if self.snn_graph_matrix[core_id][idx] > self.eps:
                    neighbors.append(idx)

        return neighbors

    def expand_clusters(self, neighbors, visited, C):
        neighbors_q = Queue()
        [neighbors_q.put(x) for x in neighbors]

        while neighbors_q.qsize() > 0:
            p = neighbors_q.get()
            if p in visited:
                continue
            else:
                self.labels[p] = C
                visited.add(p)
                new_neighbors = self.find_core_neighbors(core_id=p)
                [neighbors_q.put(x) for x in new_neighbors]

    def fit(self, features):
        if features is None:
            raise BaseException("You have to provide features argument !!!")

        self.X = features

        # Step 1
        with FuncTimer(txt_message="Distance Matrix creation time is : "):
            self.create_distance_matrix()

        # Step 2
        with FuncTimer(txt_message="Matrix sparsification time is : "):
            self.sparsify_distance_matrix()

        # Step 3
        with FuncTimer(txt_message="SNN graph construction time is : "):

            self.construct_snn_graph()

        # Step 4
        with FuncTimer(txt_message="Densities calculation time is : "):

            self.find_snn_densities()

        # Step 5
        with FuncTimer(txt_message="Core points exploration time is : "):

            self.find_core_points()

        # Step 6
        with FuncTimer(txt_message="Cluster forming time is : "):

            self.form_clusters_from_core_points()

        # Step 7
        with FuncTimer(txt_message="Noise removal time is : "):

            self.remove_noise_assign_core_points()

        return self.labels
