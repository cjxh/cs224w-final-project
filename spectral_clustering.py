import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pickle

from sklearn.cluster import SpectralClustering
from sklearn import metrics
from utils import load_pickle, dump_pickle


class NewsSpectralClustering(object):
	def __init__(self, filename, weighted=False):
		np.random.seed(1)
		if not weighted:
			self.G = nx.read_edgelist('data/{}'.format(filename))
		else:
			self.G = nx.read_weighted_edgelist('data/{}'.format(filename))
		self.adj_mat = nx.to_numpy_matrix(self.G)
		self.initialize_node_source_mappings()
		self.ground_truth = load_pickle("ground_truth.pickle")

	def calc_eigengaps(self, eigenvalues):
		eigengaps = []
		for k in range(1, len(eigenvalues)):
			eig_last = eigenvalues[k-1]
			eig_k = eigenvalues[k]
			eigengaps.append(abs(eig_k - eig_last))
		return eigengaps

	def get_optimal_k(self, eigenvalues):
		eigengaps = self.calc_eigengaps(eigenvalues)
		return np.argmax(eigengaps) + 1

	def calc_optimal_clusters(self):
		L_eigenvalues = nx.laplacian_spectrum(self.G)[:15]
		print nx.number_of_nodes(self.G)
		print L_eigenvalues
		plt.plot(np.arange(0, L_eigenvalues.shape[0]), L_eigenvalues)
		plt.show()
		return self.get_optimal_k(L_eigenvalues)

	def initialize_node_source_mappings(self):
		self.source_to_node_id_map = pickle.load(open('data/source-node-id.pickle', 'rb'))
		print self.source_to_node_id_map
		self.node_id_to_source_map = {}
		# invert source <> node_id mapping to node_id <> source mapping
		for source, id in self.source_to_node_id_map.items():
			self.node_id_to_source_map[str(id)] = source
		print self.node_id_to_source_map

	def get_trust_score_for_source(self, source):
		trust_score = None
		for key in self.ground_truth.keys():
			if key in source:
				trust_score = self.ground_truth[key]
		return trust_score

	def get_ground_truth(self):
		self.node_trust_scores_map = {}
		for node in self.G.nodes.items():
			self.node_trust_scores_map[node[0]] = self.get_trust_score_for_source(self.node_id_to_source_map[node[0]])
		print self.node_trust_scores_map

	def cluster(self, k):
		sc = SpectralClustering(k, affinity='precomputed', n_init=100)
		sc.fit(self.adj_mat)
		print('spectral clustering')
		print(sc.__dict__)
		print(sc.labels_)
		print(len(sc.labels_))
		print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
		for cluster_id in sc.labels_:
			print str(cluster_id) + ":"
			print "\t" + str([])
		print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
		print('just for better-visualization: invert clusters (permutation)')
		print(np.abs(sc.labels_ - 1))

small_network = NewsSpectralClustering('just-labeled-links.txt')
small_network.get_ground_truth()
small_network_optimal_k = small_network.calc_optimal_clusters()
small_network.cluster(small_network_optimal_k)

small_network_weighted = NewsSpectralClustering('just-labeled-links-weighted.txt', weighted=True)
small_network_weighted.get_ground_truth()
small_network_weighted_optimal_k = small_network_weighted.calc_optimal_clusters()
small_network_weighted.cluster(small_network_weighted_optimal_k)

large_network = NewsSpectralClustering('just-labeled-links.txt')
large_network.get_ground_truth()
large_network_optimal_k = large_network.calc_optimal_clusters()
large_network.cluster(large_network_optimal_k)

large_network_weighted = NewsSpectralClustering('just-labeled-links-weighted.txt', weighted=True)
large_network_weighted.get_ground_truth()
large_network_weighted_optimal_k = large_network_weighted.calc_optimal_clusters()
large_network_weighted.cluster(large_network_weighted_optimal_k)

# print('ground truth')
# print(gt)

# Cluster


# Compare ground-truth and clustering-results

# Calculate some clustering metrics
# print(metrics.adjusted_rand_score(gt, sc.labels_))
# print(metrics.adjusted_mutual_info_score(gt, sc.labels_))