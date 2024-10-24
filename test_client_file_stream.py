"""Test STT client file stream implementation in GRPC"""
from __future__ import print_function
from grpc._grpcio_metadata import __version__
import stt_pb2 as stt__pb2
import time
import stt_pb2_grpc as stt__pb2__grpc
import grpc
import argparse
from vad import (VADIterator, read_audio)
from typing import Iterator

from concurrent.futures import ThreadPoolExecutor

SAMPLING_RATE = 16000
_TIMEOUT_SECONDS_STREAM = 1000 	# timeout for streaming must be for entire stream

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
        # flow: in the the first call to the server, pass on a token, and
        # config that was returned after initial configuration. From second
        # call and later, pass on the audio chunks
        def request_stream(language_code):
            for item in generate_chunks(filename, grpc_on=True, chunkSize=chunkSize, language_code=language_code):
                yield item
        responses = service.DoSpeechToText(
            request_stream(language_code), _TIMEOUT_SECONDS_STREAM)
        executor = ThreadPoolExecutor()
        self._consumer_future = executor.submit(
            self._response_watcher, responses
        )
        print('\n\n\n\n\n\n\n\n\n\n\n\n\n\n')

    def _response_watcher(self, response_iterator: Iterator[stt__pb2.TranscriptStreamResponse]):
        for item in response_iterator:
            print(item)
        
    def createService(self, ipaddr, port):
        print(f"ip={ipaddr} port={port}")
        channel = grpc.insecure_channel(f"{ipaddr}:{port}")
        #print(type(channel))
        return stt__pb2__grpc.ListenerStub(channel)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Client to test the STT service')
    parser.add_argument('-in', action='store', dest='filename', default='temp-test-chinese.wav',
                        help='audio file')
    parser.add_argument('-a', action='store', dest='ipaddr',
                        default='localhost',
                        help='IP address of server. Default localhost.')
    parser.add_argument('-p', action='store', type=int,
                        dest='port', default=9080, help='port')
    parser.add_argument('-l', action='store', type=str,
                        dest='language', default='zh-Hans-CN', help='language')
    args = parser.parse_args()

    senderObj = Sender()
    service = senderObj.createService(args.ipaddr, args.port)
    senderObj.clientChunkStream(service, args.filename, 16000, args.language)
