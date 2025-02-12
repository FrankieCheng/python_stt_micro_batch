"""Test STT client file stream implementation in GRPC"""
from __future__ import print_function

import csv
import jiwer
from grpc._grpcio_metadata import __version__
import stt_pb2 as stt__pb2
import time
import stt_pb2_grpc as stt__pb2__grpc
import grpc
import argparse
from vad import (VADIterator, read_audio)
from typing import Iterator
from loguru import logger
from concurrent.futures import ThreadPoolExecutor
import os
from google.cloud import storage


SAMPLING_RATE = 16000
_TIMEOUT_SECONDS_STREAM = 1000 	# timeout for streaming must be for entire stream

# WAV文件的存储桶 权限call google ce
BUCKET_NAME="frankie_pc_us_central1"
# 文件前缀
TEST_FILE_PREFIX="stt/zh-CN_test_0_wav"
# WAV源内容
GROUND_TRUTH_PATH="stt/zh-cn/transcript_zh-CN_test.tsv"

# GCP服务的鉴权
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

print(f"version={__version__}")


def tensor_to_bytes(tensor):
  # Convert to NumPy (assuming 16-bit PCM audio)
    audio_np = tensor.numpy()

# Convert to bytes (little-endian for WAV format)
    byte_data = audio_np.tobytes()  # Or audio_np.tobytes(order='C') for big-endian
    return byte_data

# create an iterator that yields chunks in raw or grpc format
def generate_chunks(filename, grpc_on=False, chunkSize=3072, language_code='zh-Hans-CN'):
    if '.wav' in filename:
        all_file_chunks = read_audio(filename, sampling_rate=SAMPLING_RATE)
        index_value = 0

        while True:
            chunk = all_file_chunks[index_value *
                                    SAMPLING_RATE:index_value*SAMPLING_RATE+SAMPLING_RATE]
            index_value = index_value + 1
            bytes_list = tensor_to_bytes(chunk)

            if len(chunk) > 0:
                if grpc_on:
                    yield build_request_body(chunk=bytes_list, language_code=language_code)
                else:
                    yield chunk
            else:
                raise StopIteration
            if len(chunk) < SAMPLING_RATE:
                break
            time.sleep((int)(chunkSize/(SAMPLING_RATE - 1)))
    else:
        raise StopIteration


def build_request_body(chunk, language_code):
    return stt__pb2.SpeechChunkRequest(
        content=stt__pb2.AudioRequest(audio=chunk),
        config=stt__pb2.StreamingRecognizeRequest(
            streaming_config=stt__pb2.StreamingConfig(
                config=stt__pb2.RecognitionConfig(
                    language_codes=[language_code]
                )
            )
        )
    )


class Sender:
    def clientChunkStream(self, service, filename, chunkSize=1024, language_code='zh-Hans-CN'):
        """ send stream of chunks contaning audio bytes """

        def request_stream(language_code):
            for item in generate_chunks(filename, grpc_on=True, chunkSize=chunkSize, language_code=language_code):
                yield item
        responses = service.DoSpeechToText(
            request_stream(language_code), _TIMEOUT_SECONDS_STREAM)
        executor = ThreadPoolExecutor()
        self._consumer_future = executor.submit(
            self._response_watcher, responses
        )
        result = self._consumer_future.result()

        # logger.debug(result)

        return result

    def _response_watcher(self, response_iterator: Iterator[stt__pb2.TranscriptStreamResponse]):
        all_result = []
        for item in response_iterator:

            alternatives = item.results[0].alternatives
            if alternatives:
                try:
                    result = alternatives[0].transcript
                except:
                    result = ''
                all_result.append(result)
                # logger.debug(result)
        return ''.join(all_result)
        
    def createService(self, ipaddr, port):
        print(f"ip={ipaddr} port={port}")
        channel = grpc.insecure_channel(f"{ipaddr}:{port}")
        #print(type(channel))
        return stt__pb2__grpc.ListenerStub(channel)



def get_tsv_data():

    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(GROUND_TRUTH_PATH)
    tsv_content = blob.download_as_text()
    reader = csv.DictReader(tsv_content.splitlines(), delimiter='\t')

    rows = {row['path']:row for row in reader}

    return rows


# 设置 GCS 客户端
def download_file_from_gcs(gcs_path,local_path):
    """
    下载GCP文件到本地
    """
    gcs_path_ = gcs_path.rsplit('gs://frankie_pc_us_central1/',1)[-1]
    client = storage.Client()
    bucket = client.bucket(BUCKET_NAME)
    blob = bucket.blob(gcs_path_)
    blob.download_to_filename(local_path)

    logger.success(f"{gcs_path} 下载到本地文件 {local_path}")


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Client to test the STT service')
    # 服务端地址
    parser.add_argument('-a', action='store', dest='ipaddr',default='localhost',help='IP address of server. Default localhost.')
    # 服务端口
    parser.add_argument('-p', action='store', type=int,dest='port', default=9080, help='port')
    # 中文 zh-Hans-CN
    parser.add_argument('-l', action='store', type=str,dest='language', default='zh-Hans-CN', help='language')
    args = parser.parse_args()

    # 获取WAV文件
    files = list_files_in_prefix(BUCKET_NAME, TEST_FILE_PREFIX)
    logger.debug(f"获取到wav文件 {len(files)} 个")
    # 获取WAV源文件 内容字典 以文件名 内容 做键值对
    tsv_data = get_tsv_data()

    senderObj = Sender()
    service = senderObj.createService(args.ipaddr, args.port)

    for file in files:

        # 文件下载到本地
        file_name = file.rsplit('/')[-1]
        wav_file_mp3 = file_name.replace('.wav', '.mp3')
        local_path = os.path.join('./wav_folder',file_name)
        download_file_from_gcs(file, local_path)

        # 调用服务获取转录
        transcript_data = senderObj.clientChunkStream(service, local_path, 16000, args.language)
        logger.debug(f"文件 {file} 的识别结果:\n{transcript_data}")

        # 获取文件的原本内容
        wav_sentence = tsv_data.get(wav_file_mp3,{}).get('sentence','')
        logger.debug(f"文件 {file} 的源内容:\n{wav_sentence}")

        # 计算WER
        CER = jiwer.cer(transcript_data, wav_sentence)
        WER = jiwer.wer(transcript_data, wav_sentence)

        logger.debug(f"本次识别 CER:{CER * 100:.2f}% WER: {WER * 100:.2f}%")
        input()



# TODO 调用方式
# python3 test_client_file_stream_demo.py -a ec2-122-248-254-86.ap-southeast-1.compute.amazonaws.com -p 9080 -l zh-Hans-CN


