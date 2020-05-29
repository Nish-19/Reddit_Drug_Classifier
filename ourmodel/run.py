from utilities.datahandler import DataHandler
import logging
import argparse

if __name__ == '__main__':
	# setup the logger
	logging.basicConfig(level=logging.INFO)
	
	# instantiate data handler
	logger = logging.getLogger("Runner")

	# TODO : instantiate argparse here
	parser = argparse.ArgumentParser(description = 'input processing')
	parser.add_argument('--annotations', type = str,
						 default = 'data/annotated.csv', help = 'The path to the annotations')
	parser.add_argument('--embeddings', type = str,
						 default = 'data/reddit_emb.csv', help = 'The path to the embeddings')
	args = parser.parse_args()
	# documentation for logging https://docs.python.org/3/howto/logging.html#logging-basic-tutorial

	# TODO : check if all arguments are passed else print error 

	logger.info("Instantiating data handler")
	logger.info("Annotation path " + args.annotations + ' embeddings path ' + args.embeddings)

	# TODO : use arguments from user instead manually passing here
	#handler = DataHandler(args.annotations, args.embeddings)