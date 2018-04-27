import os
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
import numpy as np

check_wgt = 1	#1 for weighted vectors
check_model = 0	#1 for word2vec and 0 for glove
#classification algorithms

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
	clf = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes = (1,100), random_state=1)
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
	
	'''#naive-bayes multi classifier
	print("--- training naive-bayes muti classifier ---")
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
	
def main():
	
	#when to load data directly
	#np_train.seek(0)
	
	if(check_model==1):
		if (check_wgt==0):
			
			print("--- loading data ---")
			X_tr = np.load('w2v_train.npy')
			Y_tr = np.load('w2v_train_labels.npy')
	
			#np_test.seek(0)
			X_ts = np.load('w2v_test.npy')
			Y_ts = np.load('w2v_test_labels.npy')
			
			print("#training points: ",len(X_tr))
			print("#testing points: ",len(X_ts))
			
			#svm
			print("--- SVM ---")
			SVM(X_tr,Y_tr,X_ts,Y_ts)
		
			#naive bayes
			print("--- naive bayes ---")
			naive_bayes(X_tr,Y_tr,X_ts,Y_ts)
	
			#MLP
			print("--- MLP ---")
			MLP(X_tr,Y_tr,X_ts,Y_ts)
		
			#log reg
			print("--- logistic regression ---")
			log_reg(X_tr,Y_tr,X_ts,Y_ts)
		
		elif (check_wgt==1):
			print("--- loading data (weighted)---")
			X_tr = np.load('wgtw2v_train.npy')
			Y_tr = np.load('w2v_train_labels.npy')
	
			#np_test.seek(0)
			X_ts = np.load('wgtw2v_test.npy')
			Y_ts = np.load('w2v_test_labels.npy')
	
			print("#training points: ",len(X_tr))
			print("#testing points: ",len(X_ts))
			
			#svm
			print("--- SVM ---")
			SVM(X_tr,Y_tr,X_ts,Y_ts)
		
			#naive bayes
			print("--- naive bayes ---")
			naive_bayes(X_tr,Y_tr,X_ts,Y_ts)
	
			#MLP
			print("--- MLP ---")
			MLP(X_tr,Y_tr,X_ts,Y_ts)
		
			#log reg
			print("--- logistic regression ---")
			log_reg(X_tr,Y_tr,X_ts,Y_ts)	
	elif(check_model==0):
		print("--- using glove vectors ---")
		if (check_wgt==0):
			
			print("--- loading data ---")
			X_tr = np.load('glove_train.npy')
			Y_tr = np.load('glove_train_labels.npy')
	
			#np_test.seek(0)
			X_ts = np.load('glove_test.npy')
			Y_ts = np.load('glove_test_labels.npy')
			
			print("#training points: ",len(X_tr))
			print("#testing points: ",len(X_ts))
			
			#svm
			print("--- SVM ---")
			SVM(X_tr,Y_tr,X_ts,Y_ts)
		
			#naive bayes
			print("--- naive bayes ---")
			naive_bayes(X_tr,Y_tr,X_ts,Y_ts)
	
			#MLP
			print("--- MLP ---")
			MLP(X_tr,Y_tr,X_ts,Y_ts)
		
			#log reg
			print("--- logistic regression ---")
			log_reg(X_tr,Y_tr,X_ts,Y_ts)
		
		elif (check_wgt==1):
			print("--- loading data (weighted)---")
			X_tr = np.load('wgt_glove_train.npy')
			Y_tr = np.load('glove_train_labels.npy')
	
			#np_test.seek(0)
			X_ts = np.load('wgt_glove_test.npy')
			Y_ts = np.load('glove_test_labels.npy')
	
			print("#training points: ",len(X_tr))
			print("#testing points: ",len(X_ts))
			
			#svm
			print("--- SVM ---")
			SVM(X_tr,Y_tr,X_ts,Y_ts)
		
			#naive bayes
			print("--- naive bayes ---")
			naive_bayes(X_tr,Y_tr,X_ts,Y_ts)
	
			#MLP
			print("--- MLP ---")
			MLP(X_tr,Y_tr,X_ts,Y_ts)
		
			#log reg
			print("--- logistic regression ---")
			log_reg(X_tr,Y_tr,X_ts,Y_ts)	
				

if __name__ == '__main__':
	main()	
	
