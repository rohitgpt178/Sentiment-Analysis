from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
import scipy.sparse
import os
import numpy as np
import tensorflow as tf

def log_reg(X_tr,Y_tr,X_ts,Y_ts):
	print("--- training logistic regression classifier ---")
	clf = LogisticRegression()
	clf.fit(X_tr,Y_tr)
	print("--- testing logistic regression classifier ---")
	Y_pred = clf.predict(X_ts)
	acc = accuracy_score(Y_ts,Y_pred)
	print("Accuracy : ",acc*100,"%")
	
def MLP(X_tr,Y_tr,X_ts,Y_ts):
	print("--- training MLP classifier ---")
	clf = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes = (5,100), random_state=1)
	clf.fit(X_tr,Y_tr)
	print("--- testing MLP classifier ---")
	Y_pred = clf.predict(X_ts)
	acc = accuracy_score(Y_ts,Y_pred)
	print("Accuracy : ",acc*100,"%")

def SVM(X_tr,Y_tr,X_ts,Y_ts):
	print("--- training svm classifier ---")
	clf = svm.LinearSVC()
	clf.fit(X_tr,Y_tr)
	print("--- testing svm classifier ---")
	Y_pred = clf.predict(X_ts)
	acc = accuracy_score(Y_ts,Y_pred)
	print("Accuracy : ",acc*100,"%")

def naive_bayes(X_tr,Y_tr,X_ts,Y_ts):
	#naive-bayes gaussian classifier
	
	'''print("--- training naive-bayes gaussian classifier ---")
	clf = GaussianNB()
	clf.fit(X_tr,Y_tr)
	print("--- testing naive-bayes gaussian classifier ---")
	Y_pred = clf.predict(X_ts)
	acc = accuracy_score(Y_ts,Y_pred)
	print("Accuracy : ",acc*100,"%")''' 
	
	#naive-bayes multi classifier
	'''print("--- training naive-bayes muti classifier ---")
	clf = MultinomialNB()
	clf.fit(X_tr,Y_tr)
	print("--- testing naive-bayes multi classifier ---")
	Y_pred = clf.predict(X_ts)
	acc = accuracy_score(Y_ts,Y_pred)
	print("Accuracy : ",acc*100,"%")'''
	
	#naive-bayes Bernoulli classifier
	print("--- training naive-bayes Bernoulli classifier ---")
	clf = BernoulliNB()
	clf.fit(X_tr,Y_tr)
	print("--- testing naive-bayes Bernoulli classifier ---")
	Y_pred = clf.predict(X_ts)
	acc = accuracy_score(Y_ts,Y_pred)
	print("Accuracy : ",acc*100,"%")


def get_tf(X):
	tf_transformer = TfidfTransformer(use_idf=False,norm="l1").fit(X)
	return tf_transformer.transform(X)

def get_tfidf(X):
	tf_transformer = TfidfTransformer()
	return tf_transformer.fit_transform(X)
	
		
def main():
	vectorizer = CountVectorizer(binary=True)

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
	
	'''#getting bow
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
		
			
	#print(corpus)
	
	X = vectorizer.fit_transform(corpus)	#by default BOW
	
	#printf("USING tf")
	X_tf = get_tf(X)
	#printf("USING tfidf")
	X_tfidf = get_tfidf(X)
	
	X_tr = X[:n_tr]
	X_ts = X[n_tr:n_tr+n_ts]
	
	X_tr_tf = X_tf[:n_tr]
	X_ts_tf = X_tf[n_tr:n_tr+n_ts]
	
	X_tr_tfidf = X_tfidf[:n_tr]
	X_ts_tfidf = X_tfidf[n_tr:n_tr+n_ts]
	
	#print(len(vectorizer.get_feature_names()))'''
	#comment everything above while loading data
	
	#saving the bow data
	#print("Saving the training data as BOW")
	#scipy.sparse.save_npz('Xtr_sparse_bow.npz',X_tr)
	print("Loading the training data BOW")
	X_tr = scipy.sparse.load_npz('Xtr_sparse_bow.npz')
	
	#print("Saving the testing data as BOW")
	#scipy.sparse.save_npz('Xts_sparse_bow.npz',X_ts)
	print("Loading the testing data BOW")
	X_ts = scipy.sparse.load_npz('Xts_sparse_bow.npz')
	
	#saving the bow data
	#print("Saving the training data as BOW_tf")
	#scipy.sparse.save_npz('Xtr_sparse_tf.npz',X_tr_tf)
	print("Loading the training data BOW_tf")
	X_tr_tf = scipy.sparse.load_npz('Xtr_sparse_tf.npz')
	
	#print("Saving the testing data as BOW_tf")
	#scipy.sparse.save_npz('Xts_sparse_tf.npz',X_ts_tf)
	print("Loading the testing data BOW_tf")
	X_ts_tf = scipy.sparse.load_npz('Xts_sparse_tf.npz')
	
	#saving the bow data
	#print("Saving the training data as BOW_tfidf")
	#scipy.sparse.save_npz('Xtr_sparse_tfidf.npz',X_tr_tfidf)
	print("Loading the training data BOW_tfidf")
	X_tr_tfidf = scipy.sparse.load_npz('Xtr_sparse_tfidf.npz')
	
	#print("Saving the testing data as BOW_tfidf")
	#scipy.sparse.save_npz('Xts_sparse_tfidf.npz',X_ts_tfidf)
	print("Loading the testing data BOW_tfidf")
	X_ts_tfidf = scipy.sparse.load_npz('Xts_sparse_tfidf.npz')
	
	print("#training pts : ",len(X_tr.toarray()))
	print("#testing pts : ",len(X_ts.toarray()))
	
	#svm classifier
	print("--- SVM Classifier ---")
	print("--- Using BOW --------")
	SVM(X_tr,Y_tr,X_ts,Y_ts)
	print("--- Using BOW tf------")
	SVM(X_tr_tf,Y_tr,X_ts_tf,Y_ts)
	print("--- Using BOW tfidf---")
	SVM(X_tr_tfidf,Y_tr,X_ts_tfidf,Y_ts)
	
	#mlp cassifier
	print("--- MLP Classifier ---")
	print("--- Using BOW --------")
	MLP(X_tr,Y_tr,X_ts,Y_ts)
	print("--- Using BOW tf------")
	MLP(X_tr_tf,Y_tr,X_ts_tf,Y_ts)
	print("--- Using BOW tfidf---")
	MLP(X_tr_tfidf,Y_tr,X_ts_tfidf,Y_ts)
	
	#logistic regression
	print("--- Logistic Regression ---")
	print("--- Using BOW --------")
	log_reg(X_tr,Y_tr,X_ts,Y_ts)
	print("--- Using BOW tf------")
	log_reg(X_tr_tf,Y_tr,X_ts_tf,Y_ts)
	print("--- Using BOW tfidf---")
	log_reg(X_tr_tfidf,Y_tr,X_ts_tfidf,Y_ts)
	
	#naive_bayes
	print("--- Naive-Bayes classifier---")
	print("--- Using BOW --------")
	naive_bayes(X_tr,Y_tr,X_ts,Y_ts)
	print("--- Using BOW tf------")
	naive_bayes(X_tr_tf,Y_tr,X_ts_tf,Y_ts)
	print("--- Using BOW tfidf---")
	naive_bayes(X_tr_tfidf,Y_tr,X_ts_tfidf,Y_ts)
	

if __name__ == '__main__':
    main()			
