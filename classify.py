# Keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.callbacks import EarlyStopping

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelBinarizer 


import numpy as np
import io, os, gc
from keras import backend as K

from classifiers import get_multichannel_cnn_model, get_simple_rnn_model, get_lstm_model, get_rnn_cnn, get_model_lstm_atten


def get_tokenizer(vocabulary_size):
	print('Training tokenizer...')
	# train tokenizer
	all_tweets = ['data/embeddings_train01.txt', 'data/embeddings_train01.txt']
	tokenizer = Tokenizer(num_words= vocabulary_size)
	tweet_text = []
	for file_path in all_tweets:
		with io.open(file_path) as infile:	
			for line in infile:
				tweet_text.append(line.strip())
	print('Read {} tweets'.format(len(tweet_text)))
	tokenizer.fit_on_texts(tweet_text)
	return tokenizer



def get_embeddings():
	print('Generating embeddings matrix...')
	embeddings_file = 'data/reddit_emb.vec'
	embeddings_index = dict()
	with io.open(embeddings_file) as infile:
		next(infile)
		for line in infile:
			values = line.split()
			word = values[0]
			coefs = np.asarray(values[1:], dtype='float32')
			embeddings_index[word] = coefs
	# create a weight matrix for words in training docs
	vocabulary_size = len(embeddings_index)
	embeddinds_size = list(embeddings_index.values())[0].shape[0]
	print('Vocabulary = {}, embeddings = {}'.format(vocabulary_size, embeddinds_size))
	tokenizer = get_tokenizer(vocabulary_size)
	embedding_matrix = np.zeros((vocabulary_size, embeddinds_size))
	considered = 0
	total = len(tokenizer.word_index.items())
	for word, index in tokenizer.word_index.items():
		if index > vocabulary_size - 1:
			print(word, index)
			continue
		else:
			embedding_vector = embeddings_index.get(word)
			if embedding_vector is not None:
				embedding_matrix[index] = embedding_vector
				considered += 1
	print('Considered ', considered, 'Left ', total - considered)			
	return embedding_matrix, tokenizer




def get_data(tokenizer, MAX_LENGTH):
	print('Loading data')
	input_file = 'data/LIWC2015 Results (opiatesrecovery_LIWC).csv'
	
	X, Y = [], []
	with open(input_file) as infile:
		for line in infile:
			data = line.split(',')
			text, annotation = data[2], data[1]
			
			if annotation == "MET":
				X.append(text)
				Y.append("1")
			elif annotation == "Non_MET" or annotation == "Help":	
				X.append(text)
				Y.append("0")

	sequences = tokenizer.texts_to_sequences(X)
	# for i, s in enumerate(sequences):
	# 	sequences[i] = sequences[i][-250:]

	X = pad_sequences(sequences, maxlen=MAX_LENGTH)
	Y = np.array(Y)
	return X, Y	

def get_prediction(predicted):
	stacked = np.column_stack((1-predicted, predicted))
	return np.argmax(stacked,axis=1)


if __name__ == '__main__':
	embedding_matrix, tokenizer = get_embeddings()
	MAX_LENGTH = 512

	# read ml data
	X, Y = get_data(tokenizer, MAX_LENGTH)
	encoder = LabelBinarizer()#convertes into one hot form
	encoder.fit(Y)
	Y_enc = encoder.transform(Y)

	y_true, y_pred = np.array([]), np.array([])

	#k-fold evaluation
	print('Starting k-fold...')
	kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
	# Set callback functions to early stop training and save the best model so far

	fold = 0
	for train, test in kfold.split(X, Y_enc):
		unique, counts = np.unique(Y_enc[train], return_counts=True)
		print('Class distribution\n', np.asarray((unique, counts)).T)

		# model = get_lstm_model(embedding_matrix, MAX_LENGTH)
		# model = get_simple_rnn_model(embedding_matrix, MAX_LENGTH)
		# model = get_rnn_cnn(embedding_matrix, MAX_LENGTH)
		# model = get_multichannel_cnn_model(embedding_matrix, MAX_LENGTH)
		model = get_model_lstm_atten(embedding_matrix, MAX_LENGTH)
		stop = [EarlyStopping(monitor='val_loss', patience=0.001)]
		model.fit(X[train], Y_enc[train], epochs=20, batch_size=100, verbose=1, callbacks=stop)

		#Y_pred = model.predict_classes(X[test])
		Y_pred = get_prediction(model.predict(X[test]))
		temp_pred = encoder.inverse_transform(Y_pred) 
		temp_true = encoder.inverse_transform(Y_enc[test])

		print('Fold = ', fold)
		fold += 1
		print(classification_report(temp_true, temp_pred, digits=4))
		
		y_true = np.append(y_true, temp_true)
		y_pred = np.append(y_pred, temp_pred)

		del model
		gc.collect()
		K.clear_session()

	print('Overall score')
	print(classification_report(y_true, y_pred, digits=4))
	print('Accruracy')
	print(accuracy_score(y_true, y_pred))
