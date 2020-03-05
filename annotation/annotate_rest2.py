from collections import Counter	
from flask import Flask, jsonify, request, abort
import os, json, numpy
from flask_cors import CORS, cross_origin
import HTMLParser
import pandas as pd
import logging

global record_index
global data
global annotation_counter

app = Flask(__name__)
cors = CORS(app, resources={r"/annotate": {"origins": "*"}})
app.config['CORS_HEADERS'] = 'Content-Type'

def load_data():
	if os.path.isfile('annotated.csv'):
		df = pd.read_csv('annotated.csv')
	else:
		df = pd.read_csv('annotated_original.csv')	
		df = df.drop(['Primary_Class'], axis = 1)
		df.insert(1, 'Is_MET', [-1] * df.shape[0])
		df.insert(2, 'Alturism', [0] * df.shape[0])
		df.insert(3, 'Hope', [0] * df.shape[0])
		df.insert(4, 'Good_Advice', [0] * df.shape[0])
		df.insert(5, 'Bad_Advice', [0] * df.shape[0])
		df.insert(6, 'Universality', [0] * df.shape[0])
		df.to_csv('annotated.csv', index=False)
	return df

def clean(record):
	for key in record:
		if type(record[key]) == numpy.int64:
			record[key] = float(record[key])
	return record		

def get_counts():
	global data
	counts = data['Is_MET'].value_counts().to_dict()
	result = {}
	for key in counts:
		result[int(key)] = int(counts[key])
	return result

@app.route('/next', methods=['GET'])
@cross_origin(origin='*',headers=['Content-Type','Authorization'])
def get_next():
	global record_index, data
	max_records = data.shape[0]
	record_index = min(max_records, record_index + 1)
	# app.logger.info('Returning Record Index {}'.format(record_index))
	result = data.loc[record_index, ['Comment_ID', 'Is_MET', 'Alturism', 'Hope', 'Good_Advice', 'Bad_Advice', 'Universality', 'Comment']].to_dict()
	result['index'] = record_index
	result['counts'] = get_counts()
	app.logger.info(result)
	return jsonify(clean(result))

@app.route('/prev', methods=['GET'])
@cross_origin(origin='*',headers=['Content-Type','Authorization'])
def get_prev():
	global record_index, data
	record_index = max(0, record_index - 1)
	# app.logger.info('Returning Record Index {}'.format(record_index))
	result = data.loc[record_index, ['Comment_ID', 'Is_MET', 'Alturism', 'Hope', 'Good_Advice', 'Bad_Advice', 'Universality', 'Comment']].to_dict()
	result['index'] = record_index
	result['counts'] = get_counts()
	app.logger.info(result)
	return jsonify(clean(result))


@app.route('/max_records', methods=['GET'])
@cross_origin(origin='*',headers=['Content-Type','Authorization'])
def get_max_records():
	global data
	return jsonify({ 'max_records' : data.shape[0]})	


@app.route('/goto', methods=['POST'])
@cross_origin(origin='*',headers=['Content-Type','Authorization'])
def get_record():
	global record_index, data
	result = None
	index = int(request.json['index'])
	if data.shape[0] > index >= 0:
		record_index = index
		# app.logger.info('Returning Record Index {}'.format(record_index))
		result = data.loc[record_index, ['Comment_ID', 'Is_MET', 'Alturism', 'Hope', 'Good_Advice', 'Bad_Advice', 'Universality', 'Comment']].to_dict()
		result['index'] = record_index
		result['counts'] = get_counts()
		app.logger.info(result)
	else :
		abort(400)

	return jsonify(clean(result)), 200	


@app.route('/annotate', methods=['POST'])
@cross_origin(origin='*',headers=['Content-Type','Authorization'])
def annotate():
	global record_index, data, annotation_counter
	column_dict = {}
	column_dict['radioMET'] = {'column' : 'Is_MET', 'value' : 1}
	column_dict['radioNon_MET'] = {'column' : 'Is_MET', 'value' : 0}

	column_dict['radioAlturism'] = {'column' : 'Alturism', 'value' : 1}
	column_dict['radioHope'] = {'column' : 'Hope', 'value' : 1}
	column_dict['radioGoodAdvice'] = {'column' : 'Good_Advice', 'value' : 1}
	column_dict['radioBadAdvice'] = {'column' : 'Bad_Advice', 'value' : 1}
	column_dict['radioUniversality'] = {'column' : 'Universality', 'value' : 1}

	picked = column_dict[request.json['tag']]
	column = picked['column']
	value = picked['value']
	
	data.at[record_index, column] = value	
	result = data.loc[record_index, ['Comment_ID', 'Is_MET', 'Alturism', 'Hope', 'Good_Advice', 'Bad_Advice', 'Universality', 'Comment']].to_dict()
	result['index'] = record_index
	app.logger.info('Row {}, Column {} to {}'.format(record_index, column, value))
	app.logger.info(result)

	annotation_counter += 1
	if annotation_counter == 10:
		annotation_counter = 0
		app.logger.info('Saving data')
		data.to_csv('annotated.csv', index=False)


	return jsonify(clean(result)), 200



if __name__ == '__main__':
	global record_index, data
	record_index = 0
	annotation_counter = 0
	data = load_data()
	app.run(debug=True)

	
	# app.logger.warning('testing warning log')
	# app.logger.error('testing error log')
	# app.logger.info('testing info log')
