import numpy as np
import gensim
from gensim import utils
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
import nltk
from nltk import word_tokenize
from nltk import download
from nltk.corpus import stopwords
import os
from gensim.corpora import Dictionary
from gensim.models import TfidfModel

stop_words = stopwords.words('english')
print("--- loading pre-trained model ---")
#model = gensim.models.KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)
model = KeyedVectors.load_word2vec_format("gensim_glove_vectors.txt", binary=False)
print("--- DONE loading model ---")

vocab = [model.vocab]
dct = Dictionary(vocab)

def preprocess(text):
    text = text.lower()
    doc = word_tokenize(text)
    doc = [word for word in doc if word not in stop_words]
    doc = [word for word in doc if word.isalpha()] #restricts string to alphabetic characters only
    return doc

def filter_docs(corpus, labels, condition_on_doc):
	number_of_docs = len(corpus)
	labels = [i for (i, doc) in zip(labels, corpus) if condition_on_doc(doc)]
	corpus = [doc for doc in corpus if condition_on_doc(doc)]
	print("{} docs removed".format(number_of_docs - len(corpus)))
	return (corpus, labels)

def has_vector_representation(word2vec_model, doc):
	return not all(word not in word2vec_model.vocab for word in doc)

def document_vector(word2vec_model, bow_data, tfidf_model, index, doc):
	#index -> document
	wgts = []
	w2v = []
	for itr in range(0,len(tfidf_model[bow_data[index]])):
		w2v.append(word2vec_model.wv[dct[tfidf_model[bow_data[index]][itr][0]]])
		wgts.append(tfidf_model[bow_data[index]][itr][1])
	wgts = np.array(wgts)
	w2v = np.array(w2v)	
	#return np.mean(word2vec_model[doc], axis=0)
	return np.average(w2v, weights = wgts, axis=0)

def main():
	corpus = []
	n_tr = 25000;	#training points - even
	n_ts = 25000;	#testing points
	
	#labels
	y1_tr = np.zeros(int(n_tr/2));
	y2_tr = np.ones(int(n_tr/2));
	Y_tr = np.concatenate((y1_tr, y2_tr), axis=0)
	
	y1_ts = np.zeros(int(n_ts/2));
	y2_ts = np.ones(int(n_ts/2));
	Y_ts = np.concatenate((y1_ts, y2_ts), axis=0)
	
	#getting data
	dir_tr1 = os.fsencode("aclImdb/train/neg/")
	dir_tr2 = os.fsencode("aclImdb/train/pos/")
	dir_ts1 = os.fsencode("aclImdb/test/neg/")
	dir_ts2 = os.fsencode("aclImdb/test/pos/")
	
	itr = 0;
	
	for file in os.listdir(dir_tr1):
		filename = os.fsdecode(file)
		with open("aclImdb/train/neg/"+filename) as f_input:
			corpus.append(f_input.read())
		itr = itr + 1
		if itr==n_tr/2:
			break
		
	
	for file in os.listdir(dir_tr2):
		filename = os.fsdecode(file)
		with open("aclImdb/train/pos/"+filename) as f_input:
			corpus.append(f_input.read())
		itr = itr + 1
		if itr==n_tr:
			break
				
	
	itr = 0
	
	for file in os.listdir(dir_ts1):
		filename = os.fsdecode(file)
		with open("aclImdb/test/neg/"+filename) as f_input:
			corpus.append(f_input.read())
		itr = itr + 1
		if itr==n_ts/2:
			break
		
	
	for file in os.listdir(dir_ts2):
		filename = os.fsdecode(file)
		with open("aclImdb/test/pos/"+filename) as f_input:
			corpus.append(f_input.read())
		itr = itr + 1
		if itr==n_ts:
			break
	#corpus contains data

	#preprocessing
	print("--- preprocessing data ---")
	
	for itr in range(0,len(corpus)):
		corpus[itr] = preprocess(corpus[itr])	
	
	#print(corpus[0])
	
	# remove out-of-vocabulary words
	for itr in range(0,len(corpus)):
		corpus[itr] = [word for word in corpus[itr] if word in model.vocab]
	
	#getting weights - tfidf model
	
	bow_data = [dct.doc2bow(doc) for doc in corpus]
	tfidf_model = TfidfModel(bow_data)
	
	#saving tfidf models
	
	#dividing data	
	corpus_tr = corpus[:n_tr]
	corpus_ts = corpus[n_tr:n_tr+n_ts]
	#print(len(corpus_ts))
	
	#removing empty and non-useful points
	#no need for checking length of docs
	corpus_tr, Y_tr = filter_docs(corpus_tr, Y_tr, lambda doc: (len(doc) != 0))
	corpus_ts, Y_ts = filter_docs(corpus_ts, Y_ts, lambda doc: (len(doc) != 0))
	corpus_tr, Y_tr = filter_docs(corpus_tr, Y_tr, lambda doc: has_vector_representation(model, doc))
	corpus_ts, Y_ts = filter_docs(corpus_ts, Y_ts, lambda doc: has_vector_representation(model, doc))	
	
	
	#making average word2vec vectors
	print("--- getting averaged word2vec ---")
	X_tr = []
	#index = 0
	for index in range(0,len(corpus_tr)):
		doc = corpus_tr[index]
		X_tr.append(document_vector(model, bow_data, tfidf_model, index, doc))	#NOTE: index will be n_tr+ for corpus_ts
	
	X_tr = np.array(X_tr)
	#print(X_tr)
	
	print("#training points: ",len(X_tr))
		
	#saving train array
	print("saving train array")
	np.save('wgt_glove_train.npy',X_tr)
	#np.save('w2v_train_labels.npy',Y_tr)
	
	X_ts = []
	for index in range(0,len(corpus_ts)):
		doc = corpus_ts[index]
		X_ts.append(document_vector(model, bow_data, tfidf_model, index+n_tr, doc))
	X_ts = np.array(X_ts)
	
	print("#testing points: ",len(X_ts))
	
	#saving test array
	print("saving test array")
	np.save('wgt_glove_test.npy',X_ts)
	#np.save('w2v_test_labels.npy',Y_ts)
	
				

if __name__ == '__main__':
	main()
