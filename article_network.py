import snap
import pickle
from sklearn.cluster import KMeans
import sklearn
import matplotlib.pyplot as plt
import collections

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
	def __init__(self, filepath, pickled=False):
		self.node_id_to_url_map = None
		self.pickled = pickled
		if not self.pickled:
			self.graph = snap.LoadEdgeList(snap.PNGraph, filepath, 0, 1)
			self.embeddings = snap.TIntFltVH()
		else:
			self.embeddings = pickle.load(open('data/article-node2vec.pickle', 'rb'))

	def dump_pickle(self, thing_to_pickle, output=False):
		with open('data/article-node2vec.pickle', 'wb') as embeddings_pickle:
			pickle.dump(thing_to_pickle, embeddings_pickle, protocol=pickle.HIGHEST_PROTOCOL)

	def update_embeddings(self, max_k=10):
		if not self.pickled:
			snap.node2vec(self.graph, 1, 2, 128, 80, 10, 10, 1, True, self.embeddings)

		# import pdb
		# pdb.set_trace()
		X = [[i for i in self.embeddings[embedding]] for embedding in self.embeddings]
		print self.embeddings

		silhouette_scores = []
		percent_fake_per_cluster = []
		# todo: use source-node-id.pickle for article url's
		for k in range(2, max_k + 1):
			labels = KMeans(n_clusters=k)
			silhouette_scores.append((k, sklearn.metrics.silhouette_score(X, labels)))
		print(silhouette_scores)


		kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
		# import pdb
		# pdb.set_trace()
		#Default plot params
		plt.style.use('seaborn')
		cmap = 'tab10'

		plt.figure(figsize=(15,8))
		plt.subplot(121, title='"Neat" K-Means')
		plt.scatter(X[:,0], X[:,1], c=kmeans, cmap=cmap)
		print kmeans

	def trust_score_for_article(self, article_id):
		if self.node_id_to_url_map is None:
			article_url_to_node_id_map = pickle.load(open('data/article-node-id.pickle', 'rb'))
			self.node_id_to_url_map = {}
			for url, node_id in article_url_to_node_id_map.items():
				self.node_id_to_url_map[str(node_id)] = url

		article_url = self.node_id_to_url_map[article_id]
		article_url_split = article_url.split('/')
		base_url = article_url_split[2]  # .split('.')[0]
		for source in ground_truth.keys():
			if source in base_url:
				return ground_truth[source]
		return None

	def evaluate_embeddings(self, embedding_filename, max_k=20):
		embeddings = {} # node id to embedding
		with open(embedding_filename, "r") as file:
			for line in file:
				items = line.split(" ")
				embeddings[items[0]] = items[1:]

		# todo: uncomment
		silhouette_scores = []
		# for k in range(2, max_k + 1):
		# 	cluster_labels = KMeans(n_clusters=k).fit(embeddings.values())
		# 	silhouette_scores.append((k, sklearn.metrics.silhouette_score(embeddings.values(), cluster_labels.labels_)))
		# 	print(silhouette_scores[-1])
		# print(silhouette_scores)

		embeddings = collections.OrderedDict(embeddings)
		cluster_labels = KMeans(n_clusters=5).fit(embeddings.values()).labels_
		avg_trust_score_per_cluster = []
		for cluster_id in range(1, 12):

			trust_scores_for_cluster = [self.trust_score_for_article(article_id) for i, article_id in enumerate(embeddings.keys())
										if cluster_labels[i] == cluster_id]

			source_to_num_articles = {}
			total_count = 0
			for url_base in ground_truth.keys():
				count_num_articles = 0
				for index, article_id in enumerate(embeddings.keys()):
					if cluster_labels[index] != cluster_id:
						continue

					url = self.node_id_to_url_map[article_id]
					if url_base in url:
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


article_network = ArticleNetwork('data/small-snap-web-2016-09-links-clean-1.txt', pickled=False)
#article_network.update_embeddings()
article_network.evaluate_embeddings("emb/testingnode2vec.emd")
