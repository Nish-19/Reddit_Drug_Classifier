from utilities.datahandler import DataHandler
import logging
import logging.config
import argparse

if __name__ == '__main__':
	log_level = logging.INFO
	logging.basicConfig(level=log_level)
	logger = logging.getLogger("Runner")
	logger.setLevel(log_level)

	parser = argparse.ArgumentParser(description = 'input processing')
	parser.add_argument('-a', dest = 'annotations_file', type = str,
						 default = 'data/annotated.csv', help = 'The path to the annotations file. This file contains comments and the corresponding annotations.')
	parser.add_argument('-e', dest = 'embeddings_file', type = str,
						 default = 'data/reddit_emb.vec', help = 'The path to the embeddings. We use fasttext to train the word embeddings.')
	parser.add_argument('-m', dest = 'max_input_sequence_length', type = int,
						 default = '256', help = 'Specifies the maximum tokens from the comment to be consider for classification.')
	args = parser.parse_args()
	# documentation for logging https://docs.python.org/3/howto/logging.html#logging-basic-tutorial

	logger.debug("Arguments considered for current execution")
	logger.debug("Annotation path = %s" % (args.annotations_file))
	logger.debug("Embeddings path = %s" % (args.embeddings_file))
	logger.debug("Max input sequence length = %s" % (args.max_input_sequence_length))

	# instantiate data handler
	logger.info("Instantiating data handler")
	handler = DataHandler(args.annotations_file, args.embeddings_file)

	logger.info("Fetching inputs")
	tokenized_comments = handler.get_tokenized_comments(args.max_input_sequence_length)

	# loading annotations
	is_met = handler.get_met_annotations()

	# sub annotations
	is_alturism = handler.get_alturism_annotations()
	is_hope = handler.get_hope_annotations()
	is_good_advice = handler.get_good_advice_annotations()
	is_bad_advice = handler.get_bad_advice_annotations()
	is_universality = handler.get_universality_annotations()