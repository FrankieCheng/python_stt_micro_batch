import audioop
import queue

import stt_pb2 as stt__pb2
import stt_pb2_grpc as stt__pb2__grpc
import grpc
import pyaudio
import argparse
import sys
import re

# Audio recording parameters
SAMPLING_RATE = 16000
CHUNK = int(SAMPLING_RATE / 2)  # 500ms
_TIMEOUT_SECONDS_STREAM = 1000 	# timeout for streaming must be for entire stream

RED = "\033[0;31m"
GREEN = "\033[0;32m"
YELLOW = "\033[0;33m"

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

def play_audio(transcript,service):
    def request_iterator(transcript):
        for text in [transcript]:
            yield stt__pb2.TextRequest(text=text)

    responses = service.txt_to_speech(request_iterator(transcript))
    for response in responses:

        audio_content = response.audio
        # 配置音频参数
        FORMAT = pyaudio.paInt16  # 音频数据格式（16-bit PCM，常见格式）
        CHANNELS = 1  # 声道数量（1 为单声道，2 为立体声）
        RATE = 8000  # 采样率（单位：Hz）

        # 初始化 PyAudio
        audio = pyaudio.PyAudio()

        # 打开输出流
        stream = audio.open(format=FORMAT,
                            channels=CHANNELS,
                            rate=RATE,
                            output=True)

        # 播放音频流
        for i in range(0, len(audio_content), CHUNK):
            mulaw_chunk = audio_content[i:i + CHUNK]  # 取出音频块
            pcm_chunk = audioop.ulaw2lin(mulaw_chunk, 2)  # 解码为 16-bit PCM
            stream.write(pcm_chunk)  # 播放解码后的音频块

        # 关闭流和终端
        stream.stop_stream()
        stream.close()
        audio.terminate()


def listen_print_loop(responses: object, stream: object,service) -> None:
    """Iterates through server responses and prints them.

    The responses passed is a generator that will block until a response
    is provided by the server.

    Each response may contain multiple results, and each result may contain
    multiple alternatives; for details, see https://goo.gl/tjCPAU.  Here we
    print only the transcription for the top alternative of the top result.

    In this case, responses are provided for interim results as well. If the
    response is an interim one, print a line feed at the end of it, to allow
    the next result to overwrite it, until the response is a final one. For the
    final one, print a newline to preserve the finalized transcription.

    Arg:
        responses: The responses returned from the API.
        stream: The audio stream to be processed.
    """
    for response in responses:
        if response is None:
            continue

        if not response.results:
            continue

        result = response.results[0]

        if not result.alternatives:
            continue

        transcript = result.alternatives[0].transcript

        if result.is_final:
            sys.stdout.write(GREEN)
            sys.stdout.write("\033[K")
            sys.stdout.write(": " + transcript + "\n")

            stream.last_transcript_was_final = True

            play_audio(transcript, service)

            # Exit recognition if any of the transcribed phrases could be
            # one of our keywords.
            if re.search(r"\b(exit|quit)\b", transcript, re.I):
                sys.stdout.write(YELLOW)
                sys.stdout.write("Exiting...\n")
                stream.closed = True
                break
        else:
            sys.stdout.write(RED)
            sys.stdout.write("\033[K")
            sys.stdout.write(": " + transcript + "\r")

            stream.last_transcript_was_final = False

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
        
        listen_print_loop(responses, stream,service)

if __name__ == "__main__":
    main()