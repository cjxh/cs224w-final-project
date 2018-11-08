import snap
from sklearn.cluster import KMeans


class ArticleNetwork(object):
	def __init__(self, filepath):
		self.graph = snap.LoadEdgeList(snap.PNGraph, filepath, 0, 1)
		self.embeddings = snap.TIntFltVH()

	def update_embeddings(self):
		snap.node2vec(self.graph, 1, 2, 128, 80, 10, 10, 1, True, self.embeddings )
		X = [embedding for embedding in self.embeddings]
		kmeans = KMeans(n_clusters=2, random_state=0).fit(X)


article_network = ArticleNetwork('data/snap-web-2016-09-links-clean-1.txt')
article_network.update_embeddings()
