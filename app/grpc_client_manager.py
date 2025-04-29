# app/grpc_client_manager.py
import grpc
import asyncio
import logging
import base64

# Ensure these are the REGENERATED pb2 and pb2_grpc files
import stt_pb2 as stt__pb2
import stt_pb2_grpc as stt__pb2__grpc

logger = logging.getLogger(__name__)

GRPC_SERVER_ADDRESS = "localhost:53926"  # Or your gRPC server address
_TIMEOUT_SECONDS_STREAM = 1000 # As per your original client

async def generate_stt_requests_from_web_audio(audio_chunk_iterator, language_code: str):
    """
    Async generator for creating gRPC SpeechChunkRequest messages.
    It sends the full config with every audio chunk, as per the .proto definition.
    """
    try:
        async for audio_chunk_bytes in audio_chunk_iterator:
            if audio_chunk_bytes is None: # Sentinel for end of stream from WebSocket
                logger.info("Audio stream from WebSocket ended.")
                break

            # 1. Create RecognitionConfig
            recognition_cfg = stt__pb2.RecognitionConfig(
                language_codes=[language_code]
                # If your server uses sample_rate_hertz from config, set it here.
                # E.g., sample_rate_hertz=16000 (ensure this matches audio from browser)
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


async def stream_audio_to_grpc(audio_chunk_iterator, language_code: str):
    """
    Connects to the gRPC server, sends audio chunks (with config in each),
    and yields processed responses (dictionaries for the WebSocket).
    """
    grpc_channel = None
    try:
        grpc_channel = grpc.aio.insecure_channel(GRPC_SERVER_ADDRESS)
        stt_stub = stt__pb2__grpc.ListenerStub(grpc_channel)

        request_stream_generator = generate_stt_requests_from_web_audio(audio_chunk_iterator, language_code)
        logger.info(f"Initiating gRPC DoSpeechToText stream to {GRPC_SERVER_ADDRESS} for language: {language_code}")

        # response_from_grpc is stt__pb2.TranscriptStreamResponse
        async for response_from_grpc in stt_stub.DoSpeechToText(request_stream_generator, timeout=_TIMEOUT_SECONDS_STREAM):
            if not response_from_grpc:
                # logger.debug("Received an empty response_from_grpc.")
                continue

            if response_from_grpc.results:
                for result_item in response_from_grpc.results:  # result_item is stt__pb2.TranscriptStreamResult
                    if result_item.alternatives:
                        # Process the first alternative
                        alt = result_item.alternatives[0]  # alt is stt__pb2.Alternative

                        b64_audio_content = None
                        audio_format = None
                        if result_item.is_final and alt.translation and alt.synthesized_speech_audio:
                            b64_audio_content = base64.b64encode(alt.synthesized_speech_audio).decode('utf-8')
                            audio_format = "mp3" # Assuming your TTS outputs MP3

                        # Prepare data to send to the WebSocket client
                        data_for_websocket = {
                            "type": "transcription",
                            "transcript": alt.transcript,
                            "confidence": alt.confidence,
                            "translation": alt.translation if alt.translation else None,
                            "is_final": result_item.is_final,
                            "audio_data_b64": b64_audio_content, # Assuming this is populated correctly
                            "audio_format": audio_format,        # Assuming this is populated correctly
                            
                            # Add the new duration fields
                            "stt_duration_ms": alt.stt_duration_ms if hasattr(alt, 'stt_duration_ms') else None,
                            "translation_duration_ms": alt.translation_duration_ms if hasattr(alt, 'translation_duration_ms') else None,
                            "tts_duration_ms": alt.tts_duration_ms if hasattr(alt, 'tts_duration_ms') else None,
                        }
                        yield data_for_websocket
            # else:
                # logger.debug("gRPC response contained no results.")

    except grpc.aio.AioRpcError as e:
        logger.error(f"gRPC AioRpcError: Code={e.code()} Details='{e.details()}'")
        yield {"type": "error", "message": f"gRPC communication error: {e.details()}"}
    except Exception as e:
        logger.error(f"Unexpected error in stream_audio_to_grpc: {e}", exc_info=True)
        yield {"type": "error", "message": f"An unexpected server error occurred: {str(e)}"}
    finally:
        if grpc_channel:
            await grpc_channel.close()
            logger.info("gRPC channel closed.")