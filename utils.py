import cPickle as pickle


def dump_pickle(filename, data):
	with open(filename, 'wb') as pickle_file:
		pickle.dump(data, pickle_file, protocol=pickle.HIGHEST_PROTOCOL)
		print "Successfully saved data to {}.".format(filename)

'''
Returns data from filename as a python obj.
'''
def load_pickle(filename, output=False):
	with open(filename, 'rb') as pickle_file:
		data = pickle.load(pickle_file)
		if output:
			data
		print "Successfully loaded data from {}.".format(filename)
		return data

def store_ground_truth():
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
		"apnews" : 0.83,
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
	dump_pickle("ground_truth.pickle", ground_truth)
