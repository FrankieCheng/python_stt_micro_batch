# app/grpc_client_manager.py
import grpc
import asyncio
import logging
import base64

# Ensure these are the REGENERATED pb2 and pb2_grpc files
# that include 'enable_tts' in RecognitionConfig
import stt_pb2 as stt__pb2
import stt_pb2_grpc as stt__pb2__grpc

logger = logging.getLogger(__name__)

# CRITICAL: Ensure this port matches where your stt_server.py is ACTUALLY running.
# Your stt_server.py usually runs on port 9080.
# The log you showed previously had "localhost:53925" here which was strange.
# Defaulting to 9080 based on typical stt_server.py setup.
GRPC_SERVER_ADDRESS = "localhost:53925"
_TIMEOUT_SECONDS_STREAM = 1000 # As per your original client

# MODIFIED: Added enable_tts parameter
async def generate_stt_requests_from_web_audio(audio_chunk_iterator, language_code: str, enable_tts: bool):
    """
    Async generator for creating gRPC SpeechChunkRequest messages.
    It sends the full config, including the TTS preference, with every audio chunk.
    """
    logger.debug(f"generate_stt_requests_from_web_audio: TTS preference set to {enable_tts}")
    try:
        async for audio_chunk_bytes in audio_chunk_iterator:
            if audio_chunk_bytes is None: # Sentinel for end of stream from WebSocket
                logger.info("Audio stream from WebSocket ended.")
                break

            # 1. Create RecognitionConfig, now including enable_tts
            recognition_cfg = stt__pb2.RecognitionConfig(
                language_codes=[language_code],
                enable_tts=enable_tts  # <-- SET THE TTS PREFERENCE HERE
                # If your server uses sample_rate_hertz from config, set it here.
                # E.g., sample_rate_hertz=16000
            )
            # 2. Create StreamingConfig
            streaming_cfg = stt__pb2.StreamingConfig(config=recognition_cfg)
            # 3. Create StreamingRecognizeRequest (wrapper for StreamingConfig)
            streaming_recognize_req = stt__pb2.StreamingRecognizeRequest(streaming_config=streaming_cfg)
            # 4. Create AudioRequest
            audio_req = stt__pb2.AudioRequest(audio=audio_chunk_bytes)

            # 5. Create the final SpeechChunkRequest
            grpc_request = stt__pb2.SpeechChunkRequest(
                content=audio_req,
                config=streaming_recognize_req
            )
            yield grpc_request
    except Exception as e:
        logger.error(f"Error in generate_stt_requests_from_web_audio: {e}", exc_info=True)
        raise # Propagate error to the caller

# MODIFIED: Added enable_tts parameter
async def stream_audio_to_grpc(audio_chunk_iterator, language_code: str, enable_tts: bool):
    """
    Connects to the gRPC server, sends audio chunks (with config in each, including TTS pref),
    and yields processed responses (dictionaries for the WebSocket).
    """
    grpc_channel = None
    logger.info(f"Initiating gRPC DoSpeechToText stream to {GRPC_SERVER_ADDRESS} for language: {language_code}, TTS enabled: {enable_tts}")
    try:
        grpc_channel = grpc.aio.insecure_channel(GRPC_SERVER_ADDRESS)
        stt_stub = stt__pb2__grpc.ListenerStub(grpc_channel)

        # Pass enable_tts to the request generator
        request_stream_generator = generate_stt_requests_from_web_audio(audio_chunk_iterator, language_code, enable_tts)
        
        # response_from_grpc is stt__pb2.TranscriptStreamResponse
        async for response_from_grpc in stt_stub.DoSpeechToText(request_stream_generator, timeout=_TIMEOUT_SECONDS_STREAM):
            if not response_from_grpc:
                continue

            if response_from_grpc.results:
                for result_item in response_from_grpc.results:  # result_item is stt__pb2.TranscriptStreamResult
                    if result_item.alternatives:
                        alt = result_item.alternatives[0]  # alt is stt__pb2.Alternative

                        b64_audio_content = None
                        audio_format = None
                        # This part assumes the server (stt_server.py) will respect 'enable_tts'
                        # and might not send 'synthesized_speech_audio' if TTS was disabled.
                        # Or, if stt_server.py always sends it (but empty if TTS was off), this is fine.
                        if result_item.is_final and alt.translation and hasattr(alt, 'synthesized_speech_audio') and alt.synthesized_speech_audio:
                            b64_audio_content = base64.b64encode(alt.synthesized_speech_audio).decode('utf-8')
                            audio_format = "mp3"

                        data_for_websocket = {
                            "type": "transcription",
                            "transcript": alt.transcript,
                            "confidence": alt.confidence if hasattr(alt, 'confidence') else None,
                            "translation": alt.translation if hasattr(alt, 'translation') else None,
                            "is_final": result_item.is_final,
                            "audio_data_b64": b64_audio_content,
                            "audio_format": audio_format,
                            "stt_duration_ms": alt.stt_duration_ms if hasattr(alt, 'stt_duration_ms') else None,
                            "translation_duration_ms": alt.translation_duration_ms if hasattr(alt, 'translation_duration_ms') else None,
                            "tts_duration_ms": alt.tts_duration_ms if hasattr(alt, 'tts_duration_ms') else None,
                        }
                        yield data_for_websocket

    except grpc.aio.AioRpcError as e:
        logger.error(f"gRPC AioRpcError connecting to {GRPC_SERVER_ADDRESS}: Code={e.code()} Details='{e.details()}'", exc_info=True)
        yield {"type": "error", "message": f"gRPC communication error with STT service: {e.details()}"}
    except Exception as e:
        logger.error(f"Unexpected error in stream_audio_to_grpc: {e}", exc_info=True)
        yield {"type": "error", "message": f"An unexpected server error occurred: {str(e)}"}
    finally:
        if grpc_channel:
            await grpc_channel.close()
            logger.info("gRPC channel closed.")