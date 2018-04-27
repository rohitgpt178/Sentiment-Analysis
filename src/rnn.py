import numpy as np
from keras.layers import Dense, LSTM, Dropout
from keras.models import Sequential

def rnn(X_tr,Y_tr,X_ts,Y_ts):
	
	model = Sequential()
	model.add(LSTM(100,  input_shape= (1,300) ,  return_sequences=False))
	model.add(Dropout(0.2))
	model.add(Dense(1, activation='sigmoid'))
	model.compile(loss="binary_crossentropy", optimizer='rmsprop', metrics=['accuracy'])
	
	print("--- training RNN ---")
	model.fit(X_tr, Y_tr, batch_size=64, epochs=40, verbose=0, shuffle=True)
	
	print("--- testing RNN ---")
	scores = model.evaluate(X_ts, Y_ts, batch_size=64)
	print(" #Accuracy: %.2f%%" % (scores[1]*100))
	
	
def main():
	print("------------- RNN using LSTM -------------")
	print("----------------- word2vec ---------------")
	#load data here
	print("--- loading data ---")
	X_tr = np.load('w2v_train.npy')
	Y_tr = np.load('w2v_train_labels.npy')
	
	X_ts = np.load('w2v_test.npy')
	Y_ts = np.load('w2v_test_labels.npy')
		
	print("#training points: ",len(X_tr))
	print("#testing points: ",len(X_ts))
	
	X_train = np.reshape(X_tr, (X_tr.shape[0], 1, X_tr.shape[1]))
	X_test = np.reshape(X_ts, (X_ts.shape[0], 1, X_ts.shape[1]))
	
	#call rnn here
	rnn(X_train,Y_tr,X_test,Y_ts)
	
	print("------------ weighted word2vec -----------")
	print("--- loading data ---")
	X_tr = np.load('wgtw2v_train.npy')
	Y_tr = np.load('w2v_train_labels.npy')
	
	X_ts = np.load('wgtw2v_test.npy')
	Y_ts = np.load('w2v_test_labels.npy')
		
	print("#training points: ",len(X_tr))
	print("#testing points: ",len(X_ts))
	
	X_train = np.reshape(X_tr, (X_tr.shape[0], 1, X_tr.shape[1]))
	X_test = np.reshape(X_ts, (X_ts.shape[0], 1, X_ts.shape[1]))

	#call rnn here
	rnn(X_train,Y_tr,X_test,Y_ts)
	
	print("------------ glove -----------")
	print("--- loading data ---")
	X_tr = np.load('glove_train.npy')
	Y_tr = np.load('glove_train_labels.npy')
	
	X_ts = np.load('glove_test.npy')
	Y_ts = np.load('glove_test_labels.npy')
		
	print("#training points: ",len(X_tr))
	print("#testing points: ",len(X_ts))
	
	X_train = np.reshape(X_tr, (X_tr.shape[0], 1, X_tr.shape[1]))
	X_test = np.reshape(X_ts, (X_ts.shape[0], 1, X_ts.shape[1]))

	#call rnn here
	rnn(X_train,Y_tr,X_test,Y_ts)
	
	print("------------ weighted glove -----------")
	print("--- loading data ---")
	X_tr = np.load('wgt_glove_train.npy')
	Y_tr = np.load('glove_train_labels.npy')
	
	X_ts = np.load('wgt_glove_test.npy')
	Y_ts = np.load('glove_test_labels.npy')
		
	print("#training points: ",len(X_tr))
	print("#testing points: ",len(X_ts))
	
	X_train = np.reshape(X_tr, (X_tr.shape[0], 1, X_tr.shape[1]))
	X_test = np.reshape(X_ts, (X_ts.shape[0], 1, X_ts.shape[1]))

	#call rnn here
	rnn(X_train,Y_tr,X_test,Y_ts)
	

if __name__ == '__main__':
	main()
	
