import tqdm
from utils import load_pickle, dump_pickle

'''
filter out data not in ground truth (store in a pickle file)
-> get base url and check if in the ground truth set
get 1000 nodes
get 1000 neighbors of those nodes
get 1000 neighbors of neighbor nodes
'''

ground_truth = {
	"occupydemocrats.com": 0,
	"buzzfeed.com" : 0.07,
	"breitbart.com": 0.07,
	"donaldjtrump.com": 0.1,
	"infowars.com" : 0.1,
	"yahoo.com" : 0.125,
	"huffingtonpost.com" : 0.2,
	"theblaze.com" : 0.2,
	"foxnews.com" : 0.23,
	"rushlimbaugh.com" : 0.32,
	"abc.com" : 0.37,
	"msnbc.com" : 0.37,
	"drudgereport.com" : 0.39,
	"nbc.com" : 0.43,
	"cnn.com" : 0.43,
	"cbs.com" : 0.5,
	"theatlantic.com" : 0.62,
	"usatoday.com" : 0.64,
	"nytimes.com" : 0.75,
	"kansascity" : 0.76,
	"seattletimes.com" : 0.76,
	".time.com" : 0.82,
	"washingtonpost.com" : 0.83,
	"denverpost.com" : 0.83,
	"apnews.com" : 0.83,
	"politico.com" : 0.83,
	"dallasnews.com" : 0.87,
	"latimes.com" : 0.87,
	"wsj.com" : 0.9,
	"theguardian" : 0.92,
	"pbs.org" : 0.92,
	"npr.org" : 0.95,
	"bbc.com" : 0.95,
	"reuters.com" : 0.96,
	"economist.com" : 1
}

class SourceData(object):
    def __init__(self):
        self.node_id_counter = 0
        self.source_to_node_mapping = {}
        #self.ground_truth_sources = load_pickle("ground_truth.pickle").keys()
        self.ground_truth_sources = ground_truth

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
            if source in article_url:
                return source
        return article_url.split("//")[1].split("/")[0]

    def generate_node_mapping(self):
        #counter = 0
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

                if len(first_set) > 100:
                    break
                #counter += 1
               #if counter == 100:
                #    break

        print "~~~~~~~~~~~~~~~~~" + str(len(first_set))

        self.infile = open(self.filepath, 'r')
        #counter = 0
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

                    if len(third_set) > 10000:
                        break
                    # counter += 1
                    # if counter == 100:
                    #     break


td = SourceData()
td.open('data/web-2016-09-links-clean-1.txt')
td.generate_node_mapping()
td.close()
