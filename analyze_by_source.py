import snap
import pickle
from sklearn.cluster import KMeans
import sklearn
import matplotlib.pyplot as plt
import collections
import networkx as nx

ground_truth = {
	"occupydemocrats": 0,
	"buzzfeed" : 0.07,
	"breitbart": 0.07,
	"donaldjtrump": 0.1,
	"infowars" : 0.1,
	"yahoo" : 0.125,
	"huffingtonpost" : 0.2,
	"theblaze" : 0.2,
	"fox" : 0.23,
	"rushlimbaugh" : 0.32,
	"abc" : 0.37,
	"msnbc" : 0.37,
	"drudgereport" : 0.39,
	"nbc" : 0.43,
	"cnn" : 0.43,
	"cbs" : 0.5,
	"theatlantic" : 0.62,
	"usatoday" : 0.64,
	"nytimes" : 0.75,
	"kansascity" : 0.76,
	"seattletimes" : 0.76,
	"time" : 0.82,
	"washingtonpost" : 0.83,
	"denverpost" : 0.83,
	"ap" : 0.83,
	"politico" : 0.83,
	"dallasnews" : 0.87,
	"latimes" : 0.87,
	"wsj" : 0.9,
	"theguardian" : 0.92,
	"pbs" : 0.92,
	"npr" : 0.95,
	"bbc" : 0.95,
	"reuters" : 0.96,
	"economist" : 1
}

class ArticleNetwork(object):
	def __init__(self):
		self.node_id_to_source_map = None

	def trust_score_for_article(self, node_id):
		source_to_node_id_map = pickle.load(open('data/source-node-id.pickle', 'rb'))
		if self.node_id_to_source_map is None:
			self.node_id_to_source_map = {}
			for source, id in source_to_node_id_map.items():
				self.node_id_to_source_map[str(id)] = source

		source_of_node = None
		for source, id in source_to_node_id_map.items():
			if id == node_id:
				source_of_node = source

		if source_of_node in ground_truth.keys():
			return ground_truth[source_of_node]
		return None

	def evaluate_embeddings(self, embedding_filename, max_k=20):
		embeddings = {} # node id to embedding
		with open(embedding_filename, "r") as file:
			for line in file:
				items = line.split(" ")
				embeddings[str(items[0])] = [float(value) for value in items[1:]]

		# todo: uncomment
		# silhouette_scores = []
		# for k in range(2, max_k + 1):
		# 	cluster_labels = KMeans(n_clusters=k).fit(embeddings.values())
		# 	silhouette_scores.append((k, sklearn.metrics.silhouette_score(embeddings.values(), cluster_labels.labels_)))
		# 	print(silhouette_scores[-1])
		# print(silhouette_scores)

		embeddings = collections.OrderedDict(embeddings)
		cluster_labels = KMeans(n_clusters=2).fit(embeddings.values()).labels_
		avg_trust_score_per_cluster = []
		#percent_single_source = []
		for cluster_id in range(1, 12):

			trust_scores_for_cluster = [self.trust_score_for_article(str(node_id)) for i, node_id in enumerate(embeddings.keys())
										if cluster_labels[i] == cluster_id and self.trust_score_for_article(int(node_id)) is not None]

			source_to_num_articles = {}

			total_count = 0
			for url_base in ground_truth.keys():
				count_num_articles = 0
				for index, node_id in enumerate(embeddings.keys()):
					if cluster_labels[index] != cluster_id:
						continue

					source =  self.node_id_to_source_map[node_id]
					if url_base in source:
						count_num_articles += 1
				source_to_num_articles[url_base] = count_num_articles
				total_count += count_num_articles

			print(source_to_num_articles)
			print("total = {}".format(total_count))

			if len(trust_scores_for_cluster) > 0:
				avg_trust_score_per_cluster.append(sum(trust_scores_for_cluster) / len(trust_scores_for_cluster))
			else:
				avg_trust_score_per_cluster.append(0)

		plt.bar(range(1, 12), avg_trust_score_per_cluster)
		plt.show()


def evaluate_embeddings():
    # using 'data/small-by-source-snap-web-2016-09-links-clean-1.txt'
    article_network = ArticleNetwork()
    article_network.evaluate_embeddings("emb/testingnode2vec-by-source.emd")

def visualize_graph():
    graph = nx.read_edgelist('data/small-by-source-snap-web-2016-09-links-clean-1.txt')
    nx.draw_networkx(graph)
    c_values = nx.clustering(graph)
    print("avg clustering coefficient {}".format(sum(c_values.values()) / len(c_values.values())))
    plt.show()

evaluate_embeddings()
#visualize_graph()

