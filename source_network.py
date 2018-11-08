import snap
from sklearn.cluster import KMeans


class SourceNetwork(object):
    def __init__(self, filepath):
        self.graph = snap.LoadEdgeList(snap.PNGraph, filepath, 0, 1)
        self.embeddings = snap.TIntFltVH()

    def update_embeddings(self):
        snap.node2vec(self.graph, 1, 2, 128, 80, 10, 10, 1, True, self.embeddings )
        X = [embedding for embedding in self.embeddings]
        kmeans = KMeans(n_clusters=2, random_state=0).fit(X)


source_network = SourceNetwork('data/snap-source-web-2016-09-links-clean-1.txt')
source_network.update_embeddings()
