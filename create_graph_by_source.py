import tqdm
from utils import load_pickle, dump_pickle

'''
filter out data not in ground truth (store in a pickle file)
-> get base url and check if in the ground truth set
get 1000 nodes
get 1000 neighbors of those nodes
get 1000 neighbors of neighbor nodes
'''


class SourceData(object):
    def __init__(self):
        self.node_id_counter = 0
        self.source_to_node_mapping = {}
        self.ground_truth_sources = load_pickle("ground_truth.pickle").keys()

    def open(self, filepath):
        self.infile = open(filepath, 'r')
        self.filepath = filepath
        path = filepath.split('/')
        filename = path[len(path) - 1]
        self.snapfile = open('data/small-by-source-snap-' + filename, 'w')

    def close(self):
        #print self.source_to_node_mapping
        dump_pickle('data/source-node-id.pickle', self.source_to_node_mapping)
        self.infile.close()
        self.snapfile.close()

    def is_valid_source(self, article_url):
        article_url_split = article_url.split('/')
        base_url = article_url_split[2]
        for source in self.ground_truth_sources:
            if source in base_url:
                return True
        return False

    def get_node_id(self, article_url):
        source = self.article_url_to_source_url(article_url)
        if source not in self.source_to_node_mapping.keys():
            self.source_to_node_mapping[source] = self.node_id_counter
            self.node_id_counter += 1

        return self.source_to_node_mapping[source]

    def article_url_to_source_url(self, article_url):
        for source in self.ground_truth_sources:
            if article_url in source:
                return source
        return article_url.split("//")[1].split("/")[0]

    def generate_node_mapping(self):
        counter = 0
        first_set = set([])
        second_set = set([])

        third_set = set([])

        for line in tqdm.tqdm(self.infile):
            data = line.decode('utf-8').strip("\n").split("\t")

            if self.is_valid_source(data[0]):
                source_node_id = self.get_node_id(data[0])

                for i in range(2, len(data)):
                    dest_node_id = self.get_node_id(data[i])
                    first_set.add(source_node_id) # todo: this was a bug in the article version. first set should be source id
                    second_set.add(dest_node_id)
                    self.snapfile.write(str(source_node_id) + "\t" + str(dest_node_id) + "\n")

                counter += 1
                if counter == 100:
                    break

        print "~~~~~~~~~~~~~~~~~" + str(len(first_set))

        self.infile = open(self.filepath, 'r')
        counter = 0
        for line in tqdm.tqdm(self.infile):
            data = line.decode('utf-8').strip("\n").split("\t")

            if not self.is_valid_source(data[0]):
                continue

            if self.get_node_id(data[0]) in second_set:   # todo: this was another bug. should be node id
                source_node_id = self.get_node_id(data[0])

                for i in range(2, len(data)):
                    dest_node_id = self.get_node_id(data[i])
                    if dest_node_id in first_set:
                        continue

                    third_set.add(source_node_id)
                    self.snapfile.write(str(source_node_id) + "\t" + str(dest_node_id) + "\n")

                    counter += 1
                    if counter == 100:
                        break


td = SourceData()
td.open('data/web-2016-09-links-clean-1.txt')
td.generate_node_mapping()
td.close()
