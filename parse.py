import progressbar
import pickle

class TextData(object):

	def __init__(self):
		self.node_id_counter = 0
		self.article_to_node_mapping = {}

	def open(self, filepath):
		self.infile  = open(filepath, 'r')
		path = filepath.split('/')
		filename = path[len(path)-1]
		self.snapfile = open('data/snap-'+filename, 'w')
	
	def get_node_id(self, article_url):
		if article_url not in self.article_to_node_mapping.keys():
			self.article_to_node_mapping[article_url] = self.node_id_counter
			self.node_id_counter += 1

		return self.article_to_node_mapping[article_url]

	def generate_node_mapping(self):
		for line in progressbar.progressbar(self.infile):
			data = line.decode('utf-8').strip("\n").split("\t")

			source_node_id = self.get_node_id(data[0])

			for i in range(2, len(data)):
				dest_node_id = self.get_node_id(data[i])
				self.snapfile.write(str(source_node_id) + "\t" + str(dest_node_id) + "\n")

	def dump_pickle(self, output=False):
		with open('data/article-node-id.pickle', 'wb') as article_node_pickle:
			pickle.dump(self.article_to_node_mapping, article_node_pickle, protocol=pickle.HIGHEST_PROTOCOL)

	def close(self):
		self.dump_pickle(output=True)
		self.infile.close()
		self.snapfile.close()

td = TextData()
td.open('data/web-2016-09-links-clean-1.txt')
td.generate_node_mapping()
td.close()