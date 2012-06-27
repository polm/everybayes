from nltk import *
from nltk.corpus import stopwords
import sys, os, re, string, argparse, pickle


def sanitize(text):
	"""Clean the provided text.
	
	Remove stop words and punctuation and split the input into words.
	Case is preserved."""
	text = WordPunctTokenizer().tokenize(text)
	#Remove stopwords, punctuation
	bag = [t for t in set(text) if re.match('[A-z]',t) and not t in stopwords.words('english')]
	return bag

def word_bag(text):
	"""For each of our pre-determined test words we note whether or not it was present. """
	bag = sanitize(text)
	features = {}
	for w in test_words:
		features["contains:%s" % w] = (w in bag)
	return features

def file_bag(fname):
	"""Generate a word-presence feature set for the given file."""
	f = open(fname)
	text = f.read()
	f.close()
	return word_bag(text)

def build_model(path="."):
	"""Using the provided path (or .), build a model.

	The path should point to a directory containing 
	only one directory for each category. The directory 
	for a given category contains examples of that category."""
	#1. Get list of directories (categories)
	dirs = os.listdir(path)
	featuresets = []
	for d in dirs:
		fs = os.listdir(path + "/" + d)
		for f in fs:
			featuresets.append((file_bag(path + "/" +  d + "/" + f),d))
		#2. Train a model for each one
	model = NaiveBayesClassifier.train(featuresets)
	return model
	#TODO save models

def save_model(model, fname):
	f = open(fname,'wb')
	pickle.dump(model,f)
	f.close()

def load_model(fname):
	f = open(fname)
	model = pickle.load(f)
	f.close()
	return model

def load_test_words(fname):
	"""Load in the test words from the given file.

	Open and read from the provided file and return the thousand 
	most common words after removing stop words. Preserve case."""
	f = open(fname)
	raw = f.read()
	f.close()
	text = WordPunctTokenizer().tokenize(raw)
	bag = [t for t in text if re.match('[A-z]',t) and not t in stopwords.words('english')]
	return FreqDist(bag).keys()[:1000]

def identify(model,fname):
	"""Return the classification of the given file."""
	return model.classify(file_bag(fname))


def main():
	parser = argparse.ArgumentParser(description="Classification for everyone.")
	parser.add_argument('-m','--model_file',required=False,dest='model_file',default='model.pickle',help='The model file or where to save it.')
	parser.add_argument('-t','--train',required=False,dest='train',action='store_true')
	parser.add_argument('-c','--classify',required=False,dest='classify',action='store_true',default=True,help="This is the default.")
	#If learning, supply a directory with a subdirectory for each category (subdirs contain examples)
	#If classifying, put documents to classify in a directory, which is the argument
	parser.add_argument('input',nargs='+',help="The directory to learn or files to classify")
	parser.add_argument('-f','--features',required=False,nargs=1,dest='feature_list',default='features',help="The words to look for.")

	args = parser.parse_args()


	

	global test_words
	test_words = load_test_words(args.feature_list)
	if args.train:
		assert len(args.input) == 1, "Please supply a training directory."
		model = build_model(args.input[0])
		save_model(model,args.model_file)
		print "Saved model to %s" % args.model_file
		sys.exit(0)
	elif args.classify:
		model = load_model(args.model_file)
		for f in args.input:
			print f + ":" + identify(model,f)
		sys.exit(0)

if __name__ == "__main__":
	main()


test_words = load_test_words('../all.txt')
m = build_model()
print m.labels()
print m.most_informative_features()
print m.classify(file_bag('../test/fruit/peach'))
print m.classify(file_bag('../test/music/neil_young'))
print m.classify(file_bag('../test/music/sviib'))
print m.classify(file_bag('../test/fruit/grape'))
print m.classify(file_bag('../test/buildings/tokyo_tower'))
print m.classify(file_bag('../test/buildings/parthenon'))
print (file_bag('../test/buildings/parthenon') == file_bag('../test/buildings/tokyo_tower')) 

	





