import stt_pb2_grpc as stt__pb2__grpc # Your generated gRPC file
from concurrent import futures
import argparse
import logging
import asyncio
import grpc
import time

from transcribe_server import TranscriptionServer
from google.cloud import texttospeech

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - [%(funcName)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('SttServer')

class Listener(stt__pb2__grpc.ListenerServicer):
    def __init__(self, project, location, recognizer_id_arg: str) -> None:
        super().__init__()
        self.project = project
        self.location = location
        self.recognizer_id = recognizer_id_arg
        logger.info(f"[Listener] Initialized with project='{project}', location='{location}', recognizer_id='{recognizer_id_arg}'")

    async def DoSpeechToText(self, request_iterator, context: grpc.aio.ServicerContext):
        logger.info("[Listener] New DoSpeechToText call received. Creating new TranscriptionServer instance.")
        transcription_server_instance = TranscriptionServer(
            project_id=self.project,
            location=self.location,
            recognizer_id_str=self.recognizer_id
        )
        logger.info("[Listener] TranscriptionServer instance created for this call.")

        # Default TTS preference if not specified in the very first request (though it should be)
        # This is important because config is per-chunk in your proto.
        client_wants_tts = True # Default to True if not specified

        try:
            async for stt_request in request_iterator:
                audio_chunk = stt_request.content.audio
                
                # --- Read TTS Preference from Request Config ---
                # Path: stt_request.config.streaming_config.config.enable_tts
                if stt_request.config and \
                   stt_request.config.streaming_config and \
                   stt_request.config.streaming_config.config:
                    # Check if the field exists, good practice after proto changes
                    if hasattr(stt_request.config.streaming_config.config, 'enable_tts'):
                        client_wants_tts = stt_request.config.streaming_config.config.enable_tts
                        logger.debug(f"[Listener] Client TTS preference received: {client_wants_tts}")
                    else:
                        logger.warning("[Listener] 'enable_tts' field not found in RecognitionConfig. Defaulting TTS preference.")
                # --- End Read TTS Preference ---

                language_code = stt_request.config.streaming_config.config.language_codes[0]
                logger.debug(f"[Listener] Processing chunk for lang '{language_code}', size {len(audio_chunk)}, TTS Pref: {client_wants_tts}")

                response_proto = await transcription_server_instance.recv_audio_bytes(audio_chunk, language_code)

                if response_proto:
                    if response_proto.results:
                        for result_item in response_proto.results:
                            if result_item.is_final and result_item.alternatives:
                                alt = result_item.alternatives[0]
                                
                                # Initialize tts_duration_ms for the proto field
                                current_tts_duration_ms = 0

                                # Check if translation exists AND client wants TTS AND audio not already synthesized
                                if client_wants_tts and alt.translation and \
                                   (not hasattr(alt, 'synthesized_speech_audio') or not alt.synthesized_speech_audio):
                                    
                                    logger.info(f"[Listener] Client wants TTS. Final translation for TTS: '{alt.translation}'")
                                    
                                    tts_start_time = time.perf_counter()
                                    tts_mp3_response = text2speech(alt.translation)
                                    tts_end_time = time.perf_counter()
                                    current_tts_duration_ms = int((tts_end_time - tts_start_time) * 1000)
                                    
                                    if tts_mp3_response and tts_mp3_response.audio_content:
                                        logger.info(f"[Listener] TTS generated MP3 audio, length: {len(tts_mp3_response.audio_content)}, duration: {current_tts_duration_ms:.2f} ms")
                                        # Ensure fields exist on 'alt' before assignment (good after proto changes)
                                        if hasattr(alt, 'synthesized_speech_audio'):
                                            alt.synthesized_speech_audio = tts_mp3_response.audio_content
                                        if hasattr(alt, 'tts_duration_ms'):
                                            alt.tts_duration_ms = current_tts_duration_ms
                                    else:
                                        logger.warning(f"[Listener] TTS for '{alt.translation}' failed or produced no audio. TTS attempt duration: {current_tts_duration_ms:.2f} ms")
                                        if hasattr(alt, 'tts_duration_ms'):
                                             alt.tts_duration_ms = current_tts_duration_ms # Log attempt duration even on failure
                                
                                elif not client_wants_tts and alt.translation:
                                    logger.info(f"[Listener] Client has TTS disabled. Skipping TTS for translation: '{alt.translation}'")
                                    if hasattr(alt, 'tts_duration_ms'):
                                        alt.tts_duration_ms = 0 # Explicitly set to 0 if TTS skipped
                                elif alt.translation and hasattr(alt, 'synthesized_speech_audio') and alt.synthesized_speech_audio:
                                    logger.debug(f"[Listener] TTS audio already present for translation: '{alt.translation}'")
                                    # If tts_duration_ms wasn't set when audio was added, it might be 0 here.
                                    # This assumes if synthesized_speech_audio is present, tts_duration_ms was also handled.
                                elif not alt.translation:
                                     logger.debug("[Listener] No translation available for TTS.")

                    yield response_proto
                else:
                    logger.debug("[Listener] recv_audio_bytes from TranscriptionServer returned None for this chunk.")
        except Exception as e:
            logger.error(f"[Listener] Error during DoSpeechToText stream processing: {e}", exc_info=True)
            raise

# ... (serve function remains the same as your provided version) ...
async def serve(port, project, location, recognizer_id_arg: str):
    server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=20))
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
        await server.stop(0) 
        logger.info("gRPC server stopped.")

# ... (text2speech function remains the same as your provided version) ...
def text2speech(input_text: str): 
    logger.debug(f"text2speech called for: '{input_text}' (length: {len(input_text)})")
    if not input_text or not input_text.strip():
        logger.warning("text2speech called with empty or whitespace-only input.")
        return None
    try:
        client = texttospeech.TextToSpeechClient()
        synthesis_input = texttospeech.SynthesisInput(text=input_text)
        voice = texttospeech.VoiceSelectionParams(
            language_code="en-US", 
            ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL 
        )
        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.MP3
        )
        response = client.synthesize_speech(
            request={"input": synthesis_input, "voice": voice, "audio_config": audio_config}
        )
        return response
    except Exception as e:
        logger.error(f"Error in text2speech for input '{input_text[:50]}...': {e}", exc_info=True)
        return None

# ... (if __name__ == '__main__' block remains the same as your provided version) ...
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