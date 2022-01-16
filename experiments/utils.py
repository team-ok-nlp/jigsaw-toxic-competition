import os
import pandas as pd
import numpy as np
import random

from minio import Minio
from minio.error import InvalidResponseError

import torch

BASE_DIR = os.path.abspath('../')

def getData(ip='10.250.108.225', port='3011',
				userid='minioadmin', pwd='minioadmin',
				bucket='jigsaw', data_path='4th/v0/train.csv'):

	# we assume that name of data path and minio bucket path are the same.
	save_file_path = os.path.join(BASE_DIR,'data', data_path)

	# get and save data from minio server when it is not in local dir
	if not os.path.isfile(save_file_path):
		minioClient = Minio(f'{ip}:{port}', 
						access_key=userid,
						secret_key=pwd,
						secure=False)
		try:
			minioClient.fget_object(bucket, data_path, save_file_path)
			print(f"Download Complete from Minio Server.")
			print(f"in {save_file_path}")
		except InvalidResponseError as e:
			print(e)

	print(f"Read {data_path} ...")
	
	# load saved dataset in local	
	data = pd.read_csv(save_file_path)

	return data

def clean(data, col=None, is_eval=False):
	'''
	clean text in dataframe
	'''
	
	if not is_eval:
		data = data[col]

	# Clean some punctutations
	# data[col] = data[col].str.replace('\n', ' \n ')
	data = data.str.replace(r'([a-zA-Z]+)([/!?.])([a-zA-Z]+)',r'\1 \2 \3')
	# Replace repeating characters more than 3 times to length of 3
	data = data.str.replace(r'([*!?\'])\1\1{2,}',r'\1\1\1')    
	# Add space around repeating characters
	data = data.str.replace(r'([*!?\']+)',r' \1 ')    
	# patterns with repeating characters 
	data = data.str.replace(r'([a-zA-Z])\1{2,}\b',r'\1\1')
	data = data.str.replace(r'([a-zA-Z])\1\1{2,}\B',r'\1\1\1')
	data = data.str.replace(r'[ ]{2,}|\n',' ')
	# filter ibans(국제계좌형식)
	# filter email
	# filter websites
	# filter phone number
	# quotation marks
	pattern = r'(fr\d{2}[ ]\d{4}[ ]\d{4}[ ]\d{4}[ ]\d{4}[ ]\d{2}|fr\d{20}|fr[ ]\d{2}[ ]\d{3}[ ]\d{3}[ ]\d{3}[ ]\d{5})|' \
			   '((?:(?!.*?[.]{2})[a-zA-Z0-9](?:[a-zA-Z0-9.+!%-]{1,64}|)|\"[a-zA-Z0-9.+!% -]{1,64}\")@[a-zA-Z0-9][a-zA-Z0-9.-]+(.[a-z]{2,}|.[0-9]{1,}))|' \
			   '((https?:\/\/)?(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*))|' \
			   '([0-9]+.[0-9]+.[0-9]+.[0-9]+)|' \
			   '((?:(?:\+|00)33[\s.-]{0,3}(?:\(0\)[\s.-]{0,3})?|0)[1-9](?:(?:[\s.-]?\d{2}){4}|\d{2}(?:[\s.-]?\d{3}){2})|(\d{2}[ ]\d{2}[ ]\d{3}[ ]\d{3}))|' \
			   '\"'
	data = data.str.replace(pattern, '')

	data = data.dropna()
	
	return data

def set_seed(seed = 12345):
    '''Sets the seed of the entire notebook so results are the same every time we run.
    This is for REPRODUCIBILITY.'''
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)