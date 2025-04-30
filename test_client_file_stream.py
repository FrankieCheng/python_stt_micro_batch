"""Test STT client file stream implementation in GRPC"""
from __future__ import print_function
from grpc._grpcio_metadata import __version__
import stt_pb2 as stt__pb2
import time
import stt_pb2_grpc as stt__pb2__grpc
import grpc
import argparse
# from vad import (VADIterator, read_audio) # read_audio is used
from typing import Iterator
import torch # Assuming read_audio returns a torch tensor
import torchaudio # For read_audio fallback
import numpy as np # For tensor_to_bytes if tensor contains float

from concurrent.futures import ThreadPoolExecutor

SAMPLING_RATE = 16000
_TIMEOUT_SECONDS_STREAM = 1000  # timeout for streaming must be for entire stream

print(f"version={__version__}")

# Assuming read_audio function is defined elsewhere or like this simplified version
def read_audio(path: str, sampling_rate: int = 16000) -> torch.Tensor:
    """
    Reads an audio file from a given path, resamples it to the target sampling_rate,
    and converts it to mono. Returns a 1D PyTorch tensor.
    """
    try:
        sox_backends = set(['sox', 'sox_io'])
        audio_backends = torchaudio.list_audio_backends()

        if len(sox_backends.intersection(audio_backends)) > 0:
            effects = [
                ['channels', '1'],
                ['rate', str(sampling_rate)],
                ['norm'] # Added normalization, common for VAD/ASR
            ]
            waveform, sr = torchaudio.sox_effects.apply_effects_file(path, effects=effects)
        else:
            waveform, sr = torchaudio.load(path)
            if waveform.size(0) > 1: # Convert to mono by averaging channels
                waveform = waveform.mean(dim=0, keepdim=True)
            if sr != sampling_rate:
                transform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=sampling_rate)
                waveform = transform(waveform)
                sr = sampling_rate
            # Normalize
            waveform = waveform / waveform.abs().max()

        assert sr == sampling_rate, f"Expected sample rate {sampling_rate}, but got {sr}"
        return waveform.squeeze(0) # Remove channel dimension to get 1D tensor
    except Exception as e:
        print(f"Error reading audio file {path}: {e}")
        raise


def tensor_to_bytes(tensor: torch.Tensor) -> bytes:
    """
    Converts a 1D PyTorch Float32 tensor to a byte string of raw float32 values.
    This matches what `np.frombuffer(..., dtype=np.float32)` expects.
    """
    if tensor.dtype != torch.float32:
        tensor = tensor.float() # Ensure it's float32
    # Convert to NumPy array then to bytes
    audio_np = tensor.cpu().numpy() # Ensure it's on CPU
    byte_data = audio_np.astype(np.float32).tobytes()
    return byte_data

# create an iterator that yields chunks in raw or grpc format
def generate_chunks(filename: str, grpc_on: bool = False, chunk_duration_s: float = 1.0, language_code: str = 'zh-Hans-CN') -> Iterator:
    """
    Generates audio chunks from a file.
    chunk_duration_s: duration of each audio chunk in seconds.
    """
    if '.wav' in filename:
        try:
            all_file_tensor = read_audio(filename, sampling_rate=SAMPLING_RATE)
        except Exception as e:
            print(f"Stopping chunk generation due to audio read error: {e}")
            return # Stop iteration

        samples_per_chunk = int(SAMPLING_RATE * chunk_duration_s)
        current_pos = 0

        while current_pos < len(all_file_tensor):
            chunk_tensor = all_file_tensor[current_pos : current_pos + samples_per_chunk]
            current_pos += samples_per_chunk

            if chunk_tensor.numel() > 0: # Check if tensor is not empty
                bytes_list = tensor_to_bytes(chunk_tensor)
                if grpc_on:
                    yield build_request_body(chunk=bytes_list, language_code=language_code)
                else: # This branch seems unused in the current script's flow
                    yield chunk_tensor
            else: # Should not happen if len(all_file_tensor) > current_pos initially
                break 
            
            if len(chunk_tensor) < samples_per_chunk : # Last chunk might be smaller
                print(f"Last chunk generated, size: {len(chunk_tensor)} samples")
                break
            
            # Control sending rate
            # The original sleep was: time.sleep((int)(chunkSize/(SAMPLING_RATE - 1)))
            # If chunkSize was 16000 (samples), this was ~1 second.
            # Now, if chunk_duration_s is 1.0, we sleep for 1.0s.
            print(f"Generated chunk of {chunk_duration_s:.2f}s, sleeping for {chunk_duration_s:.2f}s...")
            time.sleep(chunk_duration_s) 
    else:
        print(f"File format not supported or file does not contain .wav: {filename}")
        raise StopIteration


def build_request_body(chunk: bytes, language_code: str) -> stt__pb2.SpeechChunkRequest:
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
    def clientChunkStream(self, service, filename: str, chunk_duration_s: float = 1.0, language_code: str = 'zh-Hans-CN'):
        """ send stream of chunks contaning audio bytes """
        def request_stream(lang_code: str):
            # Pass chunk_duration_s to generate_chunks
            for item in generate_chunks(filename, grpc_on=True, chunk_duration_s=chunk_duration_s, language_code=lang_code):
                yield item
        
        responses = service.DoSpeechToText(request_stream(language_code), timeout=_TIMEOUT_SECONDS_STREAM)
        
        executor = ThreadPoolExecutor(max_workers=1) # One thread for watching responses
        self._consumer_future = executor.submit(self._response_watcher, responses)
        # The many print('\n') calls are for clearing screen, less ideal than specific console clear commands.
        # For now, I'll leave them as they express user's intent to see new output clearly.
        print('\n\n\n\n\n\n\n\n\n\n\n\n\n\n')

    def _response_watcher(self, response_iterator: Iterator[stt__pb2.TranscriptStreamResponse]):
        """
        Watches for responses from the server and prints a summary.
        """
        print("<<< Waiting for responses from server... >>>\n")
        for item_index, item in enumerate(response_iterator): # item is TranscriptStreamResponse
            print(f"--- Response Batch Received #{item_index + 1} ---")
            if hasattr(item, 'speech_event_offset') and item.speech_event_offset: # Check if field is set
                 print(f"  Overall Speech Event Offset (samples): {item.speech_event_offset}")

            if not item.results:
                print("  No results in this response batch.")
            
            for result_idx, result in enumerate(item.results): # result is TranscriptStreamResult
                print(f"  Result #{result_idx + 1}:")
                print(f"    Is Final: {result.is_final}")
                if hasattr(result, 'result_end_offset') and result.result_end_offset:
                     print(f"    Result End Offset (samples): {result.result_end_offset}")
                
                if not result.alternatives:
                    print("    No alternatives in this result.")

                for alt_idx, alt in enumerate(result.alternatives): # alt is Alternative
                    print(f"    Alternative #{alt_idx + 1}:")
                    print(f"      Transcript: \"{alt.transcript}\"")
                    
                    if hasattr(alt, 'translation') and alt.translation:
                        print(f"      Translation: \"{alt.translation}\"")
                    
                    if hasattr(alt, 'confidence'): # Confidence is often present
                        print(f"      Confidence: {alt.confidence:.3f}")
                    
                    # Durations (check if fields exist, as proto might change)
                    if hasattr(alt, 'stt_duration_ms') and alt.stt_duration_ms > 0:
                        print(f"      STT Duration: {alt.stt_duration_ms} ms")
                    if hasattr(alt, 'translation_duration_ms') and alt.translation_duration_ms > 0:
                        print(f"      Translation Duration: {alt.translation_duration_ms} ms")
                    if hasattr(alt, 'tts_duration_ms') and alt.tts_duration_ms > 0:
                        print(f"      TTS Duration: {alt.tts_duration_ms} ms")

                    # Handle synthesized_speech_audio without printing all bytes
                    if hasattr(alt, 'synthesized_speech_audio') and alt.synthesized_speech_audio:
                        print(f"      Synthesized Audio: Present (Length: {len(alt.synthesized_speech_audio)} bytes)")
                    else:
                        print(f"      Synthesized Audio: Not present or empty")
                    print("     --------------------") # Separator for alternatives
            print("--- End of Response Batch ---\n")
        print("<<< Response stream finished. >>>")
        
    def createService(self, ipaddr: str, port: int) -> stt__pb2__grpc.ListenerStub:
        print(f"Connecting to gRPC server at: {ipaddr}:{port}")
        channel = grpc.insecure_channel(f"{ipaddr}:{port}")
        return stt__pb2__grpc.ListenerStub(channel)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Client to test the STT service')
    parser.add_argument('-in', action='store', dest='filename', default='temp-test-chinese.wav', help='Audio file to stream')
    parser.add_argument('-a', action='store', dest='ipaddr', default='localhost', help='IP address of server. Default localhost.')
    parser.add_argument('-p', action='store', type=int, dest='port', default=9080, help='Port of the server. Default 9080.')
    parser.add_argument('-l', action='store', type=str, dest='language', default='zh-Hans-CN', help='Language code. Default zh-Hans-CN.')
    # Added chunk duration argument
    parser.add_argument('-d', action='store', type=float, dest='duration', default=1.0, help='Duration of audio chunks to send in seconds. Default 1.0s.')

    args = parser.parse_args()

    senderObj = Sender()
    service = senderObj.createService(args.ipaddr, args.port)
    
    print(f"Streaming audio from '{args.filename}' with chunk duration {args.duration}s, language {args.language}")
    senderObj.clientChunkStream(service, args.filename, chunk_duration_s=args.duration, language_code=args.language)

    # Keep the main thread alive to allow the consumer thread to process responses
    # This is a simple way; for robust applications, use future.result() with a timeout or other eventing.
    try:
        # Wait for the _consumer_future to complete, or for a timeout/interrupt
        if senderObj._consumer_future: # Check if it was set
            senderObj._consumer_future.result(timeout=_TIMEOUT_SECONDS_STREAM + 5) # Wait a bit longer than stream timeout
    except futures.TimeoutError:
        print("Response watcher timed out.")
    except Exception as e:
        print(f"An error occurred in response watcher: {e}")
    finally:
        print("Client script finished.")