import progressbar
import pickle


class SourceTextData(object):

    def __init__(self):
        self.node_id_counter = 0
        self.source_to_node_mapping = {}

    def open(self, filepath):
        self.infile = open(filepath, 'r')
        path = filepath.split('/')
        filename = path[len(path)-1]
        self.snapfile = open('data/snap-source-'+filename, 'w')

    def get_node_id(self, article_url):
        article_url_split = article_url.split('/')
        base_url = article_url_split[2]

        if base_url not in self.source_to_node_mapping.keys():
            self.source_to_node_mapping[base_url] = self.node_id_counter
            self.node_id_counter += 1

        return self.source_to_node_mapping[base_url]

    def generate_node_mapping(self):
        for line in progressbar.progressbar(self.infile):
            data = line.decode('utf-8').strip("\n").split("\t")

            source_node_id = self.get_node_id(data[0])

            for i in range(2, len(data)):
                dest_node_id = self.get_node_id(data[i])
                self.snapfile.write(str(source_node_id) + "\t" + str(dest_node_id) + "\n")

    def dump_pickle(self, output=False):
        with open('data/source-node-id.pickle', 'wb') as article_node_pickle:
            pickle.dump(self.source_to_node_mapping, article_node_pickle, protocol=pickle.HIGHEST_PROTOCOL)

    def close(self):
        self.dump_pickle(output=True)
        self.infile.close()
        self.snapfile.close()

td = SourceTextData()
td.open('data/web-2016-09-links-clean-1.txt')
td.generate_node_mapping()
td.close()
