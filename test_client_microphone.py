
import queue

import stt_pb2 as stt__pb2
import stt_pb2_grpc as stt__pb2__grpc
import grpc
import pyaudio
import argparse

# Audio recording parameters
SAMPLING_RATE = 16000
CHUNK = int(SAMPLING_RATE / 2)  # 500ms
_TIMEOUT_SECONDS_STREAM = 1000 	# timeout for streaming must be for entire stream

class MicrophoneStream:
    """Opens a recording stream as a generator yielding the audio chunks."""

    def __init__(self: object, rate: int = SAMPLING_RATE, chunk: int = CHUNK) -> None:
        """The audio -- and generator -- is guaranteed to be on the main thread."""
        self._rate = rate
        self._chunk = chunk

        # Create a thread-safe buffer of audio data
        self._buff = queue.Queue()
        self.closed = True

    def __enter__(self: object) -> object:
        self._audio_interface = pyaudio.PyAudio()
        self._audio_stream = self._audio_interface.open(
            format=pyaudio.paFloat32,
            # The API currently only supports 1-channel (mono) audio
            # https://goo.gl/z757pE
            channels=1,
            rate=self._rate,
            input=True,
            frames_per_buffer=self._chunk,
            # Run the audio stream asynchronously to fill the buffer object.
            # This is necessary so that the input device's buffer doesn't
            # overflow while the calling thread makes network requests, etc.
            stream_callback=self._fill_buffer,
        )

        self.closed = False

        return self

    def __exit__(
        self: object,
        type: object,
        value: object,
        traceback: object,
    ) -> None:
        """Closes the stream, regardless of whether the connection was lost or not."""
        self._audio_stream.stop_stream()
        self._audio_stream.close()
        self.closed = True
        # Signal the generator to terminate so that the client's
        # streaming_recognize method will not block the process termination.
        self._buff.put(None)
        self._audio_interface.terminate()

    def _fill_buffer(
        self: object,
        in_data: object,
        frame_count: int,
        time_info: object,
        status_flags: object,
    ) -> object:
        """Continuously collect data from the audio stream, into the buffer.

        Args:
            in_data: The audio data as a bytes object
            frame_count: The number of frames captured
            time_info: The time information
            status_flags: The status flags

        Returns:
            The audio data as a bytes object
        """
        self._buff.put(in_data)
        return None, pyaudio.paContinue

    def generator(self: object) -> object:
        """Generates audio chunks from the stream of audio data in chunks.

        Args:
            self: The MicrophoneStream object

        Returns:
            A generator that outputs audio chunks.
        """
        while not self.closed:
            # Use a blocking get() to ensure there's at least one chunk of
            # data, and stop iteration if the chunk is None, indicating the
            # end of the audio stream.
            chunk = self._buff.get()
            if chunk is None:
                return
            data = [chunk]

            # Now consume whatever other data's still buffered.
            while True:
                try:
                    chunk = self._buff.get(block=False)
                    if chunk is None:
                        return
                    data.append(chunk)
                except queue.Empty:
                    break

            yield b"".join(data)

def build_request_body(chunk, language_code):
    print(f"sssxxxx {type(chunk)}")
    return stt__pb2.SpeechChunkRequest(
        content = stt__pb2.AudioRequest(audio = chunk),
        config = stt__pb2.StreamingRecognizeRequest(
            streaming_config = stt__pb2.StreamingConfig(
                config = stt__pb2.RecognitionConfig(
                    language_codes = [language_code]
                )
            )
        )
    )

def main() -> None:
    """Transcribe speech from audio file."""
    parser = argparse.ArgumentParser(description='Client to test the STT service')
    parser.add_argument('-a', action='store', dest='ipaddr',
        default='localhost',
        help='IP address of server. Default localhost.')
    parser.add_argument('-p', action='store', type=int, dest='port', default=9080, help='port')
    parser.add_argument('-l', action='store', type=str, dest='language', default='zh-Hans-CN', help='language')
    args = parser.parse_args()

    channel = grpc.insecure_channel(f"{args.ipaddr}:{args.port}")
    print(type(channel))
    service = stt__pb2__grpc.ListenerStub(channel)
    with MicrophoneStream(SAMPLING_RATE, CHUNK) as stream:
        def request_stream():
            for item in stream.generator():
                yield build_request_body(chunk=item, language_code = args.language)
        responses = service.DoSpeechToText(request_stream(), _TIMEOUT_SECONDS_STREAM)
        
        for response in responses:
            print(response)

if __name__ == "__main__":
    main()