import snap
import pickle
from sklearn.cluster import KMeans
import sklearn
import matplotlib.pyplot as plt

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


article_network = ArticleNetwork('data/snap-web-2016-09-links-clean-1.txt', pickled=False)
article_network.update_embeddings()
