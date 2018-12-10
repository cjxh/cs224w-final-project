import snap
import pickle
from sklearn.cluster import KMeans
import sklearn
import matplotlib.pyplot as plt
import collections
import networkx as nx
import numpy as np
import tsne
from sklearn.metrics.cluster import adjusted_mutual_info_score
import random

from nltk.cluster import KMeansClusterer
import nltk

ground_truth = {
	"occupydemocrats.com": 0,
	"buzzfeed.com" : 0.07,
	"breitbart.com": 0.07,
	"donaldjtrump.com": 0.1,
	"infowars.com" : 0.1,
	"yahoo.com" : 0.125,
	"huffingtonpost.com" : 0.2,
	"theblaze.com" : 0.2,
	"foxnews.com" : 0.23,
	"rushlimbaugh.com" : 0.32,
	"abc.com" : 0.37,
	"msnbc.com" : 0.37,
	"drudgereport.com" : 0.39,
	"nbc.com" : 0.43,
	"cnn.com" : 0.43,
	"cbs.com" : 0.5,
	"theatlantic.com" : 0.62,
	"usatoday.com" : 0.64,
	"nytimes.com" : 0.75,
	"kansascity" : 0.76,
	"seattletimes.com" : 0.76,
	".time.com" : 0.82,
	"washingtonpost.com" : 0.83,
	"denverpost.com" : 0.83,
	"apnews.com" : 0.83,
	"politico.com" : 0.83,
	"dallasnews.com" : 0.87,
	"latimes.com" : 0.87,
	"wsj.com" : 0.9,
	"theguardian" : 0.92,
	"pbs.org" : 0.92,
	"npr.org" : 0.95,
	"bbc.com" : 0.95,
	"reuters.com" : 0.96,
	"economist.com" : 1
}

class ArticleNetwork(object):
	def __init__(self, source_to_node_id_filename):
		self.node_id_to_source_map = None
		self.node_id_to_key_map = None
		self.source_to_node_id_filename = source_to_node_id_filename

		source_to_node_id_map = pickle.load(open(self.source_to_node_id_filename, 'rb'))
		self.node_id_to_source_map = {}
		for source, id in source_to_node_id_map.items():
			self.node_id_to_source_map[str(id)] = source

		self.node_id_to_key_map = {}
		source_to_node_id_map = pickle.load(open(self.source_to_node_id_filename, 'rb'))
		for source, id in source_to_node_id_map.items():
			for key in ground_truth.keys():
				if key in source:
					self.node_id_to_key_map[str(id)] = key

	def trust_score_for_source(self, node_id):
		key = self.node_id_to_key_map.get(node_id)
		if key is not None:
			return ground_truth[key]

		return None

	def is_labeled_source(self, url):
		for url_base in ground_truth.keys():
			if url_base in url:
				return True
		return False

	def get_embeddings(self, embedding_filename):
		embeddings = {}  # node id to embedding
		with open(embedding_filename, "r") as file:
			for i, line in enumerate(file):
				if i == 0:
					continue
				items = line.split(" ")
				node_id = str(items[0])
				if self.trust_score_for_source(node_id) is not None:
					embeddings[node_id] = [float(value) for value in items[1:]]
		return embeddings

	def evaluate_embeddings(self, embedding_filename, max_k=20):
		embeddings = self.get_embeddings(embedding_filename)

		silhouette_scores = []
		for k in range(2, max_k + 1):
			kclusterer = KMeansClusterer(k, distance=nltk.cluster.util.cosine_distance, repeats=1)
			cluster_labels = kclusterer.cluster(np.array(embeddings.values()), assign_clusters=True)
			silhouette_scores.append((k, sklearn.metrics.silhouette_score(embeddings.values(), cluster_labels)))
			print(silhouette_scores[-1])
		print(silhouette_scores)

		s_scores = sorted(silhouette_scores, reverse=True, key=lambda x: x[1])
		print("max for silhouette = {}".format(s_scores[0][0]))
		num_clusters = min(s_scores[0][0], 15)
		embeddings = collections.OrderedDict(embeddings)
		score = 0.0
		for _ in range(100000):
			kclusterer = KMeansClusterer(num_clusters, distance=nltk.cluster.util.cosine_distance) #repeats=1
			cluster_labels = kclusterer.cluster(np.array(embeddings.values()), assign_clusters=True)
			true_labels = [ 10 * round(self.trust_score_for_source(node_id), 1) for node_id in embeddings.keys()]
			score += adjusted_mutual_info_score(cluster_labels, true_labels)
		print("score = {}".format(score / 100.0))


		shuffled_score = 0.0
		for _ in range(1000):
			random.shuffle(cluster_labels)
			shuffled_score += adjusted_mutual_info_score(cluster_labels, true_labels)
		print("shuffled score = {}".format(shuffled_score / 1000))


		avg_trust_score_per_cluster = []
		for cluster_id in range(num_clusters):

			trust_scores_for_cluster = [self.trust_score_for_source(str(node_id)) for i, node_id in enumerate(embeddings.keys())
										if cluster_labels[i] == cluster_id and self.trust_score_for_source(str(node_id)) is not None]

			total_count = 0
			sources = []
			for url_base in ground_truth.keys():
				for index, node_id in enumerate(embeddings.keys()):
					if cluster_labels[index] != cluster_id:
						continue

					source =  self.node_id_to_source_map[node_id]
					if url_base in source:
						total_count += 1
						sources.append(source)

			print("total = {}".format(total_count))
			print(trust_scores_for_cluster)
			print(sources)

			if len(trust_scores_for_cluster) > 0:
				avg_trust_score_per_cluster.append(sum(trust_scores_for_cluster) / len(trust_scores_for_cluster))
			else:
				avg_trust_score_per_cluster.append(0)

		plt.bar(range(num_clusters), avg_trust_score_per_cluster)
		plt.show()


def evaluate_embeddings():
    article_network = ArticleNetwork('backup-data/just-labeled-data.pickle')
    article_network.evaluate_embeddings("emb/testingstruc2vec.emd")

def visualize_graph():

	graph = nx.read_edgelist('backup-data/large-unweighted.txt', create_using=nx.DiGraph)
	source_to_node_id_map = pickle.load(open('data/source-node-id.pickle', 'rb'))
	node_id_to_source_map = {}
	for source, id in source_to_node_id_map.items():
		if str(id) in graph.nodes():
			node_id_to_source_map[str(id)] = ground_truth[source]

	stats = []
	for node in graph.nodes():
		stats.append([node, node_id_to_source_map[str(node)], graph.degree[node]])
	print(stats)

	graph = nx.read_edgelist('backup-data/just-labeled-links-weighted-nx.txt', create_using=nx.DiGraph)
	network = ArticleNetwork('backup-data/just-labeled-data.pickle')
	embeddings = network.get_embeddings("emb/testingstruc2vec.emd")
	node_id_to_source_map = {}
	for node_id in graph.nodes():
		node_id_to_source_map[node_id] = network.trust_score_for_source(node_id)
	X = []
	colors = []
	labels = []
	node_ids = []
	for node_id, embedding in embeddings.items():
		X.append(embedding)
		node_ids.append(str(node_id))
		score = node_id_to_source_map.get(str(node_id))
		if score is None:
			colors.append("black")
			continue
		color = "green"
		label = "0.25 < Trust Score < 0.75"
		if score <= 0.25:
			color = "blue"
			label = "Trust Score <= 0.25"
		elif score >= 0.75:
			color = "red"
			label = "Trust Score >= 0.75"

		labels.append(label)
		colors.append(color)

	Y = tsne.tsne(np.array(X), 2, 50, 20.0)
	for color, label in [("red", "Trust Score >= 0.75"), ("blue", "Trust Score <= 0.25") , ("green", "0.25 < Trust Score < 0.75")]:
		x_vals = [Y[i, 0] for i in range(len(colors)) if colors[i] == color]
		y_vals =  [Y[i, 1] for i in range(len(colors)) if colors[i] == color]
		plt.scatter(x_vals, y_vals, c=color, label=label)

	Y = tsne.tsne(np.array(X), 2, 50, 20.0)
	plt.scatter(Y[:, 0], Y[:, 1], 20, c=colors, label=["Trust Score <= 0.25", "0.25 < Trust Score < 0.75", "Trust Score >= 0.75"])
	plt.legend()
	plt.legend(["blue", "green", "red"],["Trust Score <= 0.25", "0.25 < Trust Score < 0.75", "Trust Score >= 0.75"])
	plt.show()


	# Draw all of the nodes in the graph
	nx.draw_networkx(graph, labels=node_id_to_source_map)
	plt.show()
	weights = [graph[u][v]['weight'] for u, v in graph.edges()]
	nx.draw_networkx(graph, labels=node_id_to_source_map, width=weights)
	plt.show()

	# Calculate the clustering coefficient for the graph and the gnp graph
	graph = nx.Graph(graph)
	c_values = nx.clustering(graph)
	print("avg clustering coefficient {}".format(sum(c_values.values()) / len(c_values.values())))
	n = len(graph.nodes())
	e = len(graph.edges())
	rand_graph = nx.fast_gnp_random_graph(n, e * 1.0 / (0.5 * n * (n - 1)))
	c_values = nx.clustering(rand_graph)
	print("avg clustering coefficient {}".format(sum(c_values.values()) / len(c_values.values())))



evaluate_embeddings()
visualize_graph()

