import queue
import stt_pb2 as stt__pb2
import stt_pb2_grpc as stt__pb2__grpc
import grpc
import pyaudio
import argparse
import sys
import re
from google.cloud import texttospeech

import io
from pydub import AudioSegment
from pydub.playback import play
import time # <--- IMPORT TIME MODULE HERE

# Audio recording parameters
SAMPLING_RATE = 16000
CHUNK = int(SAMPLING_RATE / 2)  # 500ms
_TIMEOUT_SECONDS_STREAM = 1000  # timeout for streaming must be for entire stream

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

def listen_print_loop(responses: object, stream: object) -> None:
    """Iterates through server responses and prints them with timing for each step."""
    print(f"{YELLOW}--- Starting listen_print_loop ---{GREEN}")
    loop_start_time = time.time()

    for i, response in enumerate(responses):
        iteration_start_time = time.time()
        # print(f"{YELLOW}[Iteration {i}] Start time: {iteration_start_time:.4f}s{GREEN}")

        step_start_time = time.time()
        if response is None:
            # print(f"{YELLOW}[Iteration {i}] Step 'response is None' check: {time.time() - step_start_time:.6f}s - Skipping None response{GREEN}")
            continue
        # print(f"{YELLOW}[Iteration {i}] Step 'response is None' check: {time.time() - step_start_time:.6f}s{GREEN}")

        step_start_time = time.time()
        if not response.results:
            # print(f"{YELLOW}[Iteration {i}] Step 'response.results empty' check: {time.time() - step_start_time:.6f}s - Skipping empty results{GREEN}")
            continue
        # print(f"{YELLOW}[Iteration {i}] Step 'response.results empty' check: {time.time() - step_start_time:.6f}s{GREEN}")

        step_start_time = time.time()
        result = response.results[0]
        # print(f"{YELLOW}[Iteration {i}] Step 'access response.results[0]': {time.time() - step_start_time:.6f}s{GREEN}")

        step_start_time = time.time()
        if not result.alternatives:
            # print(f"{YELLOW}[Iteration {i}] Step 'result.alternatives empty' check: {time.time() - step_start_time:.6f}s - Skipping empty alternatives{GREEN}")
            continue
        print(f"{YELLOW}[Iteration {i}] Step 'result.alternatives empty' check: {time.time() - step_start_time:.6f}s{GREEN}")

        step_start_time = time.time()
        transcript = result.alternatives[0].transcript
        # print(f"{YELLOW}[Iteration {i}] Step 'access transcript': {time.time() - step_start_time:.6f}s{GREEN}")

        if result.is_final:
            final_processing_start_time = time.time()
            sys.stdout.write(GREEN)
            sys.stdout.write("\033[K")
            sys.stdout.write("transcription: " + transcript + "\n")
            print(f"{YELLOW}[Iteration {i}] Step 'print final transcript': {time.time() - final_processing_start_time:.6f}s{GREEN}")

            # ... inside the 'if result.alternatives[0].translation:' block ...
        if result.alternatives[0].translation:
            translation_processing_start_time = time.time()
            translation = result.alternatives[0].translation
            sys.stdout.write("translation: " + translation + "\n")
            print(f"{YELLOW}[Iteration {i}] Step 'print translation': {time.time() - translation_processing_start_time:.6f}s{GREEN}")

            # --- Play the translated audio ---
            sys.stdout.write(f"{YELLOW}Generating speech for: {translation}{GREEN}\n")

            t2s_start_time = time.time()
            # Request LINEAR16 PCM directly.
            # 24000 Hz is a common good quality rate for TTS.
            # You could also use SAMPLING_RATE (16000) if preferred.
            playback_sample_rate_hz = 24000
            speech_response, actual_sample_rate = text2speech(translation, sample_rate=playback_sample_rate_hz)
            t2s_duration = time.time() - t2s_start_time
            # Note: The 'text2speech call' time now includes network transfer of potentially larger PCM data.
            print(f"{YELLOW}[Iteration {i}] Step 'text2speech call (LINEAR16)': {t2s_duration:.6f}s{GREEN}")

            if speech_response and speech_response.audio_content:
                try:
                    sys.stdout.write(f"{YELLOW}Playing synthesized speech (PCM direct to PyAudio)...{GREEN}\n")
                    playback_pcm_start_time = time.time()

                    pa_playback = pyaudio.PyAudio() # Create a new PyAudio instance for playback

                    # Google TTS LINEAR16 is typically 16-bit signed PCM, mono.
                    # Confirm channels if necessary, but usually 1.
                    stream_playback = pa_playback.open(format=pyaudio.paInt16, # 16-bit PCM
                                            channels=1,                  # Mono
                                            rate=actual_sample_rate,     # Sample rate from TTS
                                            output=True)

                    stream_playback.write(speech_response.audio_content) # Write raw bytes to stream

                    stream_playback.stop_stream()
                    stream_playback.close()
                    pa_playback.terminate()

                    playback_duration = time.time() - playback_pcm_start_time
                    # This duration is how long pyaudio took to accept the data and for it to play out (blocking).
                    print(f"{YELLOW}[Iteration {i}] Step 'play PCM with PyAudio': {playback_duration:.6f}s{GREEN}")
                    sys.stdout.write(f"{YELLOW}Playback finished.{GREEN}\n")
                except Exception as e:
                    print(f"{RED}Error playing audio: {e}{GREEN}")
            print(f"{YELLOW}[Iteration {i}] Total translation audio processing: {time.time() - translation_processing_start_time:.6f}s{GREEN}")
# --- End Play audio ---

            stream.last_transcript_was_final = True

            exit_check_start_time = time.time()
            if re.search(r"\b(exit|quit)\b", transcript, re.I):
                sys.stdout.write(YELLOW)
                sys.stdout.write("Exiting...\n")
                stream.closed = True
                # print(f"{YELLOW}[Iteration {i}] Step 'exit keyword check': {time.time() - exit_check_start_time:.6f}s - Exiting{GREEN}")
                break
            else:
                print(f"{YELLOW}[Iteration {i}] Step 'exit keyword check': {time.time() - exit_check_start_time:.6f}s{GREEN}")
            print(f"{YELLOW}[Iteration {i}] Total final result processing: {time.time() - final_processing_start_time:.6f}s{GREEN}")

        else: # Interim result
            interim_processing_start_time = time.time()
            sys.stdout.write(RED)
            sys.stdout.write("\033[K")
            sys.stdout.write(": " + transcript + "\r")
            stream.last_transcript_was_final = False
            # print(f"{YELLOW}\r[Iteration {i}] Step 'print interim transcript': {time.time() - interim_processing_start_time:.6f}s (interim){GREEN}") # \r to overwrite if needed

        iteration_duration = time.time() - iteration_start_time
        print(f"{YELLOW}[Iteration {i}] Total iteration time: {iteration_duration:.4f}s{GREEN}")
        sys.stdout.flush() # Ensure timing logs are printed immediately

    total_loop_duration = time.time() - loop_start_time
    print(f"{YELLOW}--- listen_print_loop finished. Total time: {total_loop_duration:.4f}s ---{GREEN}")
    
def text2speech(input_text, sample_rate=16000): # Added sample_rate, 24kHz is good for many voices
    client = texttospeech.TextToSpeechClient()
    synthesis_input = texttospeech.SynthesisInput(text=input_text)
    # Ensure the language_code matches the language of input_text
    # If 'translation' is always English, "en-US" is fine.
    voice = texttospeech.VoiceSelectionParams(
        language_code="en-US",
        ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
    )
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.LINEAR16, # Request raw PCM
        sample_rate_hertz=sample_rate # Specify the desired sample rate
    )
    response = client.synthesize_speech(
      input=synthesis_input, voice=voice, audio_config=audio_config
    )
    # Return the actual sample rate used, as some voices might have constraints
    # However, for LINEAR16, the API should honor sample_rate_hertz.
    # For simplicity, we'll assume it's what we requested.
    return response, sample_rate

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
    # print(type(channel)) # You might want to remove or comment this out for cleaner timing logs
    service = stt__pb2__grpc.ListenerStub(channel)
    with MicrophoneStream(SAMPLING_RATE, CHUNK) as stream:
        def request_stream():
            for item in stream.generator():
                yield build_request_body(chunk=item, language_code = args.language)
        
        print(f"{YELLOW}--- Calling service.DoSpeechToText ---{GREEN}")
        service_call_start_time = time.time()
        responses = service.DoSpeechToText(request_stream(), _TIMEOUT_SECONDS_STREAM)
        print(f"{YELLOW}--- service.DoSpeechToText call returned (streaming setup time): {time.time() - service_call_start_time:.4f}s ---{GREEN}")
        
        listen_print_loop(responses, stream)

if __name__ == "__main__":
    main()