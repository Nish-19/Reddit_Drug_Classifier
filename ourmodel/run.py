from utilities.datahandler import DataHandler
import logging

if __name__ == '__main__':
	# setup the logger
	logging.basicConfig(level=logging.INFO)
	
	# instantiate data handler
	logger = logging.getLogger("Runner")

	# TODO : instantiate argparse here
	# documentation for logging https://docs.python.org/3/howto/logging.html#logging-basic-tutorial

	# TODO : check if all arguments are passed else print error 

	logger.info("Instantiating data handler")

	# TODO : use arguments from user instead manually passing here
	handler = DataHandler('data/annotated.csv', 'data/reddit_emb.vec')