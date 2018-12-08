import snap
import pickle
from sklearn.cluster import KMeans
import sklearn
import matplotlib.pyplot as plt
import collections
import networkx as nx
import numpy as np
import tsne

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

	def evaluate_embeddings(self, embedding_filename, max_k=30):
		embeddings = self.get_embeddings(embedding_filename)

		# todo: uncomment
		# silhouette_scores = []
		# for k in range(2, max_k + 1):
		# 	# cluster_labels = KMeans(n_clusters=k).fit(embeddings.values())
		# 	kclusterer = KMeansClusterer(k, distance=nltk.cluster.util.cosine_distance, repeats=1)
		# 	cluster_labels = kclusterer.cluster(np.array(embeddings.values()), assign_clusters=True)
		# 	silhouette_scores.append((k, sklearn.metrics.silhouette_score(embeddings.values(), cluster_labels)))
		# 	print(silhouette_scores[-1])
		# print(silhouette_scores)

		num_clusters = 9
		embeddings = collections.OrderedDict(embeddings)
		kclusterer = KMeansClusterer(num_clusters, distance=nltk.cluster.util.cosine_distance, repeats=1)
		cluster_labels = kclusterer.cluster(np.array(embeddings.values()), assign_clusters=True)
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
    # using 'data/small-by-source-snap-web-2016-09-links-clean-1.txt'
    article_network = ArticleNetwork('data/source-node-id.pickle')
    article_network.evaluate_embeddings("emb/testingnode2vec-by-source.emd")

def visualize_graph():

	graph = nx.read_edgelist('backup-data/just-labeled-links-weighted-nx.txt', create_using=nx.DiGraph)
	source_to_node_id_map = pickle.load(open('backup-data/just-labeled-data.pickle', 'rb'))
	node_id_to_source_map = {}
	for source, id in source_to_node_id_map.items():
		if str(id) in graph.nodes():
			node_id_to_source_map[str(id)] = ground_truth[source]

	stats = []
	for node in graph.nodes():
		stats.append([node, node_id_to_source_map[str(node)], graph.degree[node]])
	print(stats)

	network = ArticleNetwork('data/source-node-id.pickle')
	embeddings = network.get_embeddings("emb/testingnode2vec-by-source.emd")
	X = []
	colors = []
	for node_id, embedding in embeddings.items():
		X.append(embedding)
		score = node_id_to_source_map.get(str(node_id))
		if score is None:
			colors.append("black")
			continue
		color = "green"
		if score <= 0.25:
			color = "blue"
		elif score >= 0.75:
			color = "red"

		colors.append(color)
	Y = tsne.tsne(np.array(X), 2, 50, 20.0)
	plt.scatter(Y[:, 0], Y[:, 1], 20, c=colors)
	plt.show()

	weights = [graph[u][v]['weight'] for u, v in graph.edges()]
	nx.draw_networkx(graph, labels=node_id_to_source_map, width=weights) # note: can remove width option if you want
	c_values = nx.clustering(graph)
	print("avg clustering coefficient {}".format(sum(c_values.values()) / len(c_values.values())))
	plt.show()

#evaluate_embeddings()
visualize_graph()

