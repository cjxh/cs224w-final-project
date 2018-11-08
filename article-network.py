import snap


class ArticleNetwork(object):
	def __init__(self, filepath):
		self.graph = snap.LoadEdgeList(snap.PNGraph, filepath, 0, 1)
		embeddings = snap.TIntFltVH()
		snap.node2vec(self.graph, 1, 2, 128, 80, 10, 10, 1, True, embeddings)


article_network = ArticleNetwork('data/snap-web-2016-09-links-clean-1.txt')
