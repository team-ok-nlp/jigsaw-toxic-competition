from minio import Minio
# from minio.error import ResponseError
from minio.error import InvalidResponseError
import argparse


def upload_file(mc:Minio, bucket_name, object_name, file_path):
    '''upload the file on Minio Server
    '''
    try:
        mc.fput_object(bucket_name, object_name, file_path)
        print(f'{object_name} is uploaded in the bucket "{bucket_name}".')
    except InvalidResponseError as e:
        print(e)

def download_file(mc:Minio, bucket_name, object_name, file_path):
    '''download the file from Minio Server
    '''
    try:
        mc.fget_object(bucket_name, object_name, file_path)
        print(f'{object_name} is downloaded from the bucket "{bucket_name}".')
    # except ResponseError as e:
    except InvalidResponseError as e:
        print(e)

FUNCTIONS = {
    "upload" : upload_file,
    "download" : download_file
}

if __name__ == '__main__':
    # get the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--ip', '-i', type=str, default='10.250.108.225', help='ip address')
    parser.add_argument('--port', type=str, default='3011',help='minio server port')
    parser.add_argument('--userid', type=str, default='minioadmin',help='user ID of your account in S3 service')
    parser.add_argument('--password', type=str, default='minioadmin',help='password of your account in S3 service')
    parser.add_argument('--bucket_name', '-b', type=str, required=True,help='minio bucket name')
    parser.add_argument('--object_name', '-o', type=str, required=True, help='object name')
    parser.add_argument('--file_path', '-p', type=str, required=True, help='file path')
    parser.add_argument('--function', '-f', type=str, required=True, help='upload or download')
    
    args = parser.parse_args()

    # create a Minio instance
    mc = Minio(endpoint=f'{args.ip}:{args.port}',
               access_key=args.userid,
               secret_key=args.password,
               secure=False
               )

    if not mc.bucket_exists(args.bucket_name):
        raise Exception(f'{args.bucket_name} does not exist.')

    # select the function
    function = FUNCTIONS[args.function]

    # execute the function
    function(mc=mc, 
             bucket_name=args.bucket_name,
             object_name=args.object_name,
             file_path=args.file_path)