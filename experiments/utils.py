import os
import pandas as pd

from minio import Minio
from minio.error import InvalidResponseError

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
			print(f"{save_file_path}")
		except InvalidResponseError as e:
			print(e)

	# load saved dataset in local	
	data = pd.read_csv(save_file_path)

	return data

def clean(data, col):
    '''
    clean text in dataframe
    '''
    # Clean some punctutations
    # data[col] = data[col].str.replace('\n', ' \n ')
    data[col] = data[col].str.replace(r'([a-zA-Z]+)([/!?.])([a-zA-Z]+)',r'\1 \2 \3')
    # Replace repeating characters more than 3 times to length of 3
    data[col] = data[col].str.replace(r'([*!?\'])\1\1{2,}',r'\1\1\1')    
    # Add space around repeating characters
    data[col] = data[col].str.replace(r'([*!?\']+)',r' \1 ')    
    # patterns with repeating characters 
    data[col] = data[col].str.replace(r'([a-zA-Z])\1{2,}\b',r'\1\1')
    data[col] = data[col].str.replace(r'([a-zA-Z])\1\1{2,}\B',r'\1\1\1')
    data[col] = data[col].str.replace(r'[ ]{2,}|\n',' ')
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
    data[col] = data[col].str.replace(pattern, '')
    
    return data
