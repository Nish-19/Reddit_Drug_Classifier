# Contains utility functions re read text files and generate the necessary inputs required for the model
# Each dataset will be accompanied by a embedding matrix and a tokenizer, 
# hence we create a class as an abstraction to hold all these entities

import numpy as np
import logging, io
from keras.preprocessing.text import Tokenizer

# Helper class to process all the necessary text files and embeddings saved disk. 
# Use these to generate embedding matrix and tokenizer

class DataHandler:

	def __init__(self, met_annotations_file, embeddings_file):
		self.met_annotations_file = met_annotations_file
		self.embeddings_file = embeddings_file

		self.logger = logging.getLogger(self.__class__.__name__)

		# read comments and annotations
		annotated_rows = self.__get_annotated_index()

		self.logger.info("Reading annotated comments")
		comments = self.__read_columns_from_annotation([7], filter_rows = annotated_rows, data_type = np.str)
		
		self.logger.info("Reading met annotations")
		is_met = self.__read_columns_from_annotation([1], filter_rows = annotated_rows, data_type = np.int)

		self.logger.info("Reading sub-annotations")
		is_alturism = self.__read_columns_from_annotation([2], filter_rows = annotated_rows, data_type = np.int)
		is_hope = self.__read_columns_from_annotation([3], filter_rows = annotated_rows, data_type = np.int)
		is_good_advice = self.__read_columns_from_annotation([4], filter_rows = annotated_rows, data_type = np.int)
		is_bad_advice = self.__read_columns_from_annotation([5], filter_rows = annotated_rows, data_type = np.int)
		is_universality = self.__read_columns_from_annotation([6], filter_rows = annotated_rows, data_type = np.int)

		
		# read embeddings
		self.logger.info("Reading embeddings file")
		embeddings_index = self.__read_saved_word_embeddings()

		# fit tokenizer
		self.logger.info("Training the tokenizer on comments")
		vocabulary_size = len(embeddings_index)
		self.tokenizer = self.__fit_tokenizer(comments, vocabulary_size) 


		# generate embedding matrix
		self.logger.info("Generting embedding matrix")
		self.embedding_matrix = self.__generate_embedding_matrix(embeddings_index)


	# reads comments from the file and returns a list of comments
	# used to fit the tokenizer
	def __read_columns_from_annotation(self, columns, filter_rows = None, data_type = None):
		column_names = self.__get_column_names(columns)
		self.logger.debug("Reading columns (%s) from %s" % (column_names, self.met_annotations_file))
		data_type = np.float if data_type == None else data_type
		data = np.loadtxt(self.met_annotations_file, delimiter=',', skiprows=1, usecols=columns, dtype=data_type)

		

		if filter_rows is not None:
			self.logger.debug("Filtering columns (%s) down to %d rows" % (column_names, filter_rows.shape[0]))
			data = data[filter_rows]
		else:
			self.logger.debug("Read data dimentions are %s" % (data.shape))	

		return data


	# returns the indices of annotated rows. Is_MET column with values 0 or 1 indicate annotaion. Rows wiht -1 indicated unannotated
	def __get_annotated_index(self):
		self.logger.debug("Fetching annotated comments")
		annotations = self.__read_columns_from_annotation([1])
		annotated_rows = np.where(annotations >= 0)[0]
		self.logger.debug("Fetched %d annotated comments" % (annotated_rows.shape[0]))	
		return annotated_rows


	# reads first row which contans column names
	def __get_column_names(self, columns):
		column_names = []
		with open(self.met_annotations_file) as infile:
			header = next(infile).split(',')
			column_names = [header[i] for i in columns]
		return ','.join(column_names)


	# reads the contents of the saved embedding file. We use fasttext for word embeddings
	def __read_saved_word_embeddings(self):
		self.logger.debug('Reading contents of embedding file %s' % (self.embeddings_file))
		embeddings_index = dict()
		with io.open(self.embeddings_file) as infile:
			next(infile)
			for line in infile:
				values = line.split()
				word = values[0]
				coefs = np.asarray(values[1:], dtype='float32')
				embeddings_index[word] = coefs
		return embeddings_index



	# train the tokernizer to map words to a unique index	
	def __fit_tokenizer(self, comments, vocabulary_size):
		self.logger.debug('Training tokenizer')
		all_tweets = ['data/embeddings_train01.txt', 'data/embeddings_train01.txt']
		tokenizer = Tokenizer(num_words = vocabulary_size)
		tokenizer.fit_on_texts(comments)
		self.logger.debug('Tokenized {} comments'.format(len(comments)))
		return tokenizer	

	# generate a 2d matix of embeddings, where row number indicates index of the word
	def __generate_embedding_matrix(self, embeddings_index):
		vocabulary_size = len(embeddings_index)
		embeddinds_size = list(embeddings_index.values())[0].shape[0]
		self.logger.debug('Generating embeddings matrix with vocabulary size = %d, embedding dimention = %s' % (vocabulary_size, embeddinds_size))
		
		embedding_matrix = np.zeros((vocabulary_size, embeddinds_size))
		considered = 0
		total = len(self.tokenizer.word_index.items())
		for word, index in self.tokenizer.word_index.items():
			if index > vocabulary_size - 1:
				continue
			else:
				embedding_vector = embeddings_index.get(word)
				if embedding_vector is not None:
					embedding_matrix[index] = embedding_vector
					considered += 1
		self.logger.debug("Considered %d word and ignored %d words" % (considered, total - considered))		
		return embedding_matrix
