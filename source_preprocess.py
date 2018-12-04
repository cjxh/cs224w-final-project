import tqdm
import pickle


class SourceTextData(object):

    def __init__(self):
        self.node_id_counter = 0
        self.source_to_node_mapping = {}
        self.edge_weights = {}

    def open(self, filepath):
        self.infile = open(filepath, 'r')
        path = filepath.split('/')
        filename = path[len(path)-1].split(".")[0]
        self.snapfile = open('data/snap-source-'+filename+'.paj', 'w')

    def get_node_id(self, article_url):
        article_url_split = article_url.split('/')
        base_url = article_url_split[2]

        if base_url not in self.source_to_node_mapping.keys():
            self.source_to_node_mapping[base_url] = self.node_id_counter
            self.node_id_counter += 1

        return self.source_to_node_mapping[base_url]

    def create_one_set(self):
        for i in range(2, len(data)):
            dest_node_id = self.get_node_id(data[i])
            if (source_node_id, dest_node_id) in self.edge_weights:
                self.edge_weights[(source_node_id, dest_node_id)] += 1
            else:
                self.edge_weights[(source_node_id, dest_node_id)] = 1
                counter += 1

    def generate_node_mapping(self):
        counter = 0
        for line in tqdm(self.infile):
            data = line.decode('utf-8').strip("\n").split("\t")

            source_node_id = self.get_node_id(data[0])

            for i in range(2, len(data)):
                dest_node_id = self.get_node_id(data[i])
                if (source_node_id, dest_node_id) in self.edge_weights:
                    self.edge_weights[(source_node_id, dest_node_id)] += 1
                else:
                    self.edge_weights[(source_node_id, dest_node_id)] = 1
                    counter += 1

            if counter == 1000:
                break

        for source_node_id, dest_node_id in self.edge_weights.keys():
            self.snapfile.write("{}\t{}\t{}\n".format(str(source_node_id),
                                               str(dest_node_id),
                                               str(self.edge_weights[(source_node_id, dest_node_id)])))


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
