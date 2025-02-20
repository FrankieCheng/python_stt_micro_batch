from loguru import logger
from google.cloud import storage
import sys
import os
import csv


from google.cloud import speech
BUCKET_NAME="frankie_pc_us_central1"
TEST_FILE_PREFIX="stt/zh-CN_test_0_wav"
GROUND_TRUTH_PATH="stt/zh-cn/transcript_zh-CN_test.tsv"


os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/wangjie/Desktop/cloudplus1-test01-04b6f67fe50f.json"



def list_files_in_prefix(bucket_name:str, prefix:str):
    """列出 GCS 中指定 bucket 和 prefix 下的所有文件。

    Args:
        bucket_name: GCS 桶名称。
        prefix: 文件名前缀。

    Returns:
        包含所有文件名的列表。
    """
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    blobs = bucket.list_blobs(prefix=prefix)
    file_names = [f"gs://{bucket_name}/{blob.name}" for blob in blobs]
    return file_names

def get_tsv_data():

    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(GROUND_TRUTH_PATH)
    tsv_content = blob.download_as_text()
    reader = csv.DictReader(tsv_content.splitlines(), delimiter='\t')

    rows = {row['path']:row for row in reader}

    return rows


def get_stt_server_result(file):

    commond = f"python3 test_client_file_stream.py -a ec2-122-248-254-86.ap-southeast-1.compute.amazonaws.com -p 9080 -l zh-Hans-CN -in "

    result = ''

    return result



if __name__ == '__main__':
    tsv_data = get_tsv_data()

    files = list_files_in_prefix(BUCKET_NAME,TEST_FILE_PREFIX)
    # logger.debug(len(files))
    # input()
    for file in files:

        file_name = file.rsplit('/')[-1]
        wav_file_mp3 = file_name.replace('.wav', '.mp3')

        logger.debug(file)
        logger.debug(wav_file_mp3)
        logger.debug(tsv_data[wav_file_mp3])
        input()
        transcript_result = get_stt_server_result(file)

        tsv_data[wav_file_mp3]['transcript'] = transcript_result

        input()
    logger.debug(files)