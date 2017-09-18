from keras.preprocessing import sequence
from keras.optimizers import SGD, RMSprop, Adagrad
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
from keras.datasets import imdb


import pandas as pd  
from random import random

import numpy as np

import sys
sys.path.append(".")

def main():
	flow = (list(range(1,10,1)) + list(range(10,1,-1)))*1000  
	pdata = pd.DataFrame({"a":flow, "b":flow})  
	pdata.b = pdata.b.shift(9)  
	data = pdata.iloc[10:] * random()  # some noise  
	(X_train, y_train), (X_test, y_test) = train_test_split(data)  # retrieve data
	print "load data done"

	in_out_neurons = 2  
	hidden_neurons = 300
	
	max_features = 1000
 
	
	model = Sequential()  
	model.add(Embedding(max_features, 128))
	model.add(LSTM(in_out_neurons, hidden_neurons, return_sequences=False))  
	model.add(Dense(hidden_neurons, in_out_neurons))  
	model.add(Activation("linear"))  
	model.compile(loss="mean_squared_error", optimizer="rmsprop")  


	


	
	# and now train the model
	# batch_size should be appropriate to your memory size
	# number of epochs should be higher for real world problems
	#model.fit(X_train, y_train, batch_size=450, nb_epoch=10, validation_split=0.05)  
	
	
	batch_size = 32
	model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=4, validation_data=(X_test, y_test), show_accuracy=True)



	predicted = model.predict(X_test)  
	rmse = np.sqrt(((predicted - y_test) ** 2).mean(axis=0))

	# and maybe plot it
	pd.DataFrame(predicted[:100]).plot()  
	pd.DataFrame(y_test[:100]).plot()  



def _load_data(data, n_prev = 100):  
    """
    data should be pd.DataFrame()
    """

    docX, docY = [], []
    for i in range(len(data)-n_prev):
        docX.append(data.iloc[i:i+n_prev].as_matrix())
        docY.append(data.iloc[i+n_prev].as_matrix())
    alsX = np.array(docX)
    alsY = np.array(docY)

    return alsX, alsY

def train_test_split(df, test_size=0.1):  
    """
    This just splits data to training and testing parts
    """
    ntrn = round(len(df) * (1 - test_size))
    ntrn = int(ntrn)
    X_train, y_train = _load_data(df.iloc[0:ntrn])
    X_test, y_test = _load_data(df.iloc[ntrn:])

    return (X_train, y_train), (X_test, y_test)
	
if __name__ == '__main__':
	main()