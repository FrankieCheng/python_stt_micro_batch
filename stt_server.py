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
        self.transcription_server_instance = None # Will be created per call

    async def DoSpeechToText(self, request_iterator, context: grpc.aio.ServicerContext):
        logger.info("[Listener] New DoSpeechToText call received.")
        
        # Determine language_code from the first request for TranscriptionServer initialization
        # This assumes language_code and TTS pref don't change mid-stream.
        # If they can, this logic needs adjustment.
        first_request = None
        try:
            first_request = await request_iterator.__anext__()
        except StopAsyncIteration:
            logger.info("[Listener] Request iterator was empty.")
            return
        
        language_code = first_request.config.streaming_config.config.language_codes[0]
        client_wants_tts = True # Default
        if first_request.config and first_request.config.streaming_config and first_request.config.streaming_config.config:
            if hasattr(first_request.config.streaming_config.config, 'enable_tts'):
                client_wants_tts = first_request.config.streaming_config.config.enable_tts

        transcription_server_instance = TranscriptionServer(
            project_id=self.project,
            location=self.location,
            recognizer_id_str=self.recognizer_id,
            language_code=language_code # Pass language_code to TranscriptionServer
        )
        transcription_server_instance.start_processing_pipeline()
        logger.info(f"[Listener] TranscriptionServer instance created and pipeline started for lang '{language_code}', TTS pref: {client_wants_tts}")

        async def audio_feeder():
            """Feeds audio from gRPC request_iterator to TranscriptionServer's input queue."""
            try:
                # Process the first chunk already read
                await transcription_server_instance.submit_audio_chunk(first_request.content.audio)

                async for stt_request in request_iterator:
                    if transcription_server_instance._stop_event.is_set(): break
                    await transcription_server_instance.submit_audio_chunk(stt_request.content.audio)
                logger.info("[Listener] Audio feeder: All audio chunks submitted.")
            except grpc.aio.AioRpcError as e: # Catch gRPC specific errors like client cancelling stream
                logger.warning(f"[Listener] Audio feeder: gRPC error in request stream: {e.code()} - {e.details()}")
            except Exception as e:
                logger.error(f"[Listener] Audio feeder: Error: {e}", exc_info=True)
            finally:
                # Signal end of audio to the pipeline
                await transcription_server_instance.submit_audio_chunk(None) 
                logger.info("[Listener] Audio feeder: Sent None sentinel to TranscriptionServer.")
        
        feeder_task = asyncio.create_task(audio_feeder())
        
        try:
            async for response_proto in transcription_server_instance.get_ordered_results():
                if response_proto and response_proto.results:
                    # TTS logic for results coming from transcription_server_instance
                    for result_item in response_proto.results:
                        if result_item.is_final and result_item.alternatives:
                            alt = result_item.alternatives[0]
                            current_tts_duration_ms = 0
                            if client_wants_tts and alt.translation and \
                               (not hasattr(alt, 'synthesized_speech_audio') or not alt.synthesized_speech_audio):
                                logger.info(f"[Listener] Client wants TTS. Final translation for TTS: '{alt.translation}'")
                                tts_start_time = time.perf_counter()
                                # text2speech should ideally be async or run in executor to not block this result loop
                                # For simplicity, keeping it sync as per user's code for now.
                                tts_mp3_response = text2speech(alt.translation) 
                                tts_end_time = time.perf_counter()
                                current_tts_duration_ms = int((tts_end_time - tts_start_time) * 1000)
                                if tts_mp3_response and tts_mp3_response.audio_content:
                                    alt.synthesized_speech_audio = tts_mp3_response.audio_content
                                if hasattr(alt, 'tts_duration_ms'): alt.tts_duration_ms = current_tts_duration_ms
                            elif not client_wants_tts and alt.translation:
                                if hasattr(alt, 'tts_duration_ms'): alt.tts_duration_ms = 0
                    yield response_proto
                elif response_proto is None and transcription_server_instance._stop_event.is_set(): # Explicit break if needed
                     break
        except Exception as e:
            logger.error(f"[Listener] Error consuming results from TranscriptionServer: {e}", exc_info=True)
            # context.abort(grpc.StatusCode.INTERNAL, "Error processing results") # Option
            raise
        finally:
            logger.info("[Listener] Result consumption loop finished. Cleaning up.")
            await transcription_server_instance.stop_processing_pipeline()
            if feeder_task and not feeder_task.done():
                feeder_task.cancel()
            logger.info("[Listener] DoSpeechToText call ended.")


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