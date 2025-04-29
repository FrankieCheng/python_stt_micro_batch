import stt_pb2_grpc as stt__pb2__grpc # Your generated gRPC file
from concurrent import futures
import argparse
import logging
import asyncio
import grpc
import time # <--- IMPORT THE TIME MODULE

# Ensure transcribe_server.py is in the same directory or PYTHONPATH
from transcribe_server import TranscriptionServer
from google.cloud import texttospeech # For your TTS function

# Corrected and more detailed logging configuration
logging.basicConfig(
    level=logging.INFO, # Set to logging.DEBUG for maximum verbosity during troubleshooting
    format='%(asctime)s - %(levelname)s - %(name)s - [%(funcName)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('SttServer') # Logger for this specific file

class Listener(stt__pb2__grpc.ListenerServicer):
    # __init__ now accepts recognizer_id and stores it
    def __init__(self, project, location, recognizer_id_arg: str) -> None:
        super().__init__()
        self.project = project
        self.location = location
        self.recognizer_id = recognizer_id_arg # Store the recognizer ID
        logger.info(f"[Listener] Initialized with project='{project}', location='{location}', recognizer_id='{recognizer_id_arg}'")
        # IMPORTANT: Moved TranscriptionServer instantiation into DoSpeechToText
        # to ensure each call gets a fresh, state-isolated instance.

    async def DoSpeechToText(self, request_iterator, context: grpc.aio.ServicerContext):
        logger.info("[Listener] New DoSpeechToText call received. Creating new TranscriptionServer instance.")
        # Create a new TranscriptionServer instance for each call to isolate its state (like self.all_chunks)
        transcription_server_instance = TranscriptionServer(
            project_id=self.project,
            location=self.location,
            recognizer_id_str=self.recognizer_id # Pass the stored recognizer_id
        )
        logger.info("[Listener] TranscriptionServer instance created for this call.")

        try:
            async for stt_request in request_iterator:
                audio_chunk = stt_request.content.audio
                # Config is sent with every chunk as per your proto
                language_code = stt_request.config.streaming_config.config.language_codes[0]
                logger.debug(f"[Listener] Processing chunk for lang '{language_code}', size {len(audio_chunk)}")

                # Use the correct instance variable: transcription_server_instance
                response_proto = await transcription_server_instance.recv_audio_bytes(audio_chunk, language_code)

                if response_proto: # response_proto is stt__pb2.TranscriptStreamResponse
                    if response_proto.results:
                        for result_item in response_proto.results:
                            if result_item.is_final and result_item.alternatives:
                                alt = result_item.alternatives[0]
                                # Check if translation exists and if TTS has not already been done (e.g., by transcribe_server)
                                if alt.translation and not alt.synthesized_speech_audio:
                                    logger.info(f"[Listener] Final translation for TTS: '{alt.translation}'")
                                    
                                    # --- Start TTS Timing ---
                                    tts_start_time = time.perf_counter()
                                    tts_mp3_response = text2speech(alt.translation)
                                    tts_end_time = time.perf_counter()
                                    tts_duration_ms = (tts_end_time - tts_start_time) * 1000
                                    # --- End TTS Timing ---
                                    
                                    if tts_mp3_response and tts_mp3_response.audio_content:
                                        logger.info(f"[Listener] TTS generated MP3 audio, length: {len(tts_mp3_response.audio_content)}, duration: {tts_duration_ms:.2f} ms")
                                        alt.synthesized_speech_audio = tts_mp3_response.audio_content
                                    else:
                                        logger.warning(f"[Listener] TTS for '{alt.translation}' failed or produced no audio. TTS attempt duration: {tts_duration_ms:.2f} ms")
                                elif alt.translation and alt.synthesized_speech_audio:
                                    logger.debug(f"[Listener] TTS audio already present for translation: '{alt.translation}'")
                                elif not alt.translation:
                                     logger.debug("[Listener] No translation available for TTS.")

                    yield response_proto
                else:
                    # Changed from warning to debug as it can be normal if no speech segment is finalized yet
                    logger.debug("[Listener] recv_audio_bytes from TranscriptionServer returned None for this chunk.")
                    pass # Continue to next audio chunk
        except Exception as e:
            logger.error(f"[Listener] Error during DoSpeechToText stream processing: {e}", exc_info=True)
            raise 


# serve function now accepts recognizer_id
async def serve(port, project, location, recognizer_id_arg: str):
    server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=20))
    # Pass recognizer_id_arg when creating Listener
    stt__pb2__grpc.add_ListenerServicer_to_server(
        Listener(project, location, recognizer_id_arg),
        server
    )
    server.add_insecure_port(f'[::]:{port}')
    logger.info(f"gRPC server starting on port {port}...")
    await server.start()
    logger.info("gRPC server started successfully and awaiting termination.")
    try:
        await server.wait_for_termination()
    except KeyboardInterrupt:
        logger.info("gRPC server stopping due to KeyboardInterrupt...")
    finally:
        await server.stop(0) # Graceful stop
        logger.info("gRPC server stopped.")


def text2speech(input_text: str): # TTS function, ensure it's robust
    logger.debug(f"text2speech called for: '{input_text}' (length: {len(input_text)})")
    if not input_text or not input_text.strip():
        logger.warning("text2speech called with empty or whitespace-only input.")
        return None
        
    try:
        # It's generally recommended to create clients once if possible,
        # but creating per call is simpler if calls are infrequent.
        # For high throughput, consider initializing client in Listener.__init__
        # or making it global (with care for thread safety if not using gRPC stubs directly).
        client = texttospeech.TextToSpeechClient()
        synthesis_input = texttospeech.SynthesisInput(text=input_text)
        # Consider making language_code for TTS configurable if translations are not always to English
        voice = texttospeech.VoiceSelectionParams(
            language_code="en-US", 
            ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL 
        )
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3
        )
        response = client.synthesize_speech(
            request={"input": synthesis_input, "voice": voice, "audio_config": audio_config} # Using request dict
        )
        return response
    except Exception as e:
        logger.error(f"Error in text2speech for input '{input_text[:50]}...': {e}", exc_info=True) # Log only first 50 chars
        return None # Return None or an empty response object on error


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SpeechToText service')
    parser.add_argument('-p', action='store', dest='port', type=int, default=9080, help='Port to listen on.')
    parser.add_argument('-project', action='store', dest='project', type=str, required=True, help='Google Cloud Project ID.')
    parser.add_argument('-location', action='store', dest='location', type=str, default='global', help='Google Cloud Location (e.g., us-central1, global for some services).')
    parser.add_argument('-recognizer', action='store', dest='recognizer_id', type=str, default='_',
                        help='Specific Recognizer ID to use (e.g., from Speech-to-Text v2). Use "_" or leave empty for default model behavior.')
    args = parser.parse_args()

    logger.info(f"Attempting to start STT server with: port={args.port}, project='{args.project}', location='{args.location}', recognizer_id='{args.recognizer_id}'")
    asyncio.run(serve(args.port, args.project, args.location, args.recognizer_id))