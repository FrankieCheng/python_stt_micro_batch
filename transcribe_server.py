import logging
import asyncio
import functools # For functools.partial
import io
import base64
import torch
import torchaudio
import numpy as np
import json
from datetime import datetime

from google.api_core.client_options import ClientOptions
from google.cloud.speech_v2 import SpeechClient
from google.cloud.speech_v2.types import cloud_speech # RecognitionConfig, StreamingRecognitionConfig etc. are here
# from google.cloud.speech_v2.types import ExplicitDecodingConfig, StreamingRecognizeResponse -> ExplicitDecodingConfig is under cloud_speech
from google.auth.exceptions import DefaultCredentialsError


import vertexai
from vertexai.generative_models import GenerativeModel, Part
# import vertexai.preview.generative_models as generative_models # No longer need preview if using stable GenerativeModel

import stt_pb2 as stt__pb2 # Your protobuf definitions
from vad import VADIterator # Assuming these are your local VAD utilities
from utils_vad import get_speech_timestamps

# Global model name from your code
asr_model_name_gemini = "gemini-1.5-flash-002" # Renamed for clarity
TARGET_LANGUAGE = 'English'

FORMAT = '%(asctime)s - %(levelname)s - %(name)s - [%(funcName)s] - %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger('TranscriptionServer')

LANGUAGE_CODE_DIC = {
    'ar-EG':'Arabic', 'zh-Hans-CN':'Chinese', 'cmn-Hant-TW':'Traditional Chinese',
    'nl-NL':'Dutch', 'en-US':'English', 'fr-FR':'French', 'de-DE':'German',
    'hi-IN':'Hindi', 'it-IT':'Italian', 'ja-JP':'Japanese', 'pt-PT':'Portuguese',
    'es-ES':'Spanish'
}

# --- It's good practice to put the connectivity test outside the class or as a static method ---
def perform_google_cloud_connectivity_tests(project_id, location):
    logger.info("[ConnectivityTest] Starting Google Cloud connectivity tests...")
    speech_v2_ok = False
    vertex_ai_ok = False

    # Test Speech-to-Text v2 Client
    try:
        endpoint = f"{location}-speech.googleapis.com"
        logger.info(f"[ConnectivityTest] Attempting to create SpeechClient (v2) with endpoint: {endpoint}...")
        # speech_v2_test_client = SpeechClient(client_options={"api_endpoint": endpoint}) # Simpler way to pass options
        speech_v2_test_client = SpeechClient(client_options=ClientOptions(api_endpoint=endpoint))
        logger.info("[ConnectivityTest] SpeechClient (v2) created successfully for test.")
        # Optional: A very lightweight, non-mutating call, e.g., list operations if the API supports it without complex params
        # For now, successful client instantiation is a good sign against "Connection Refused".
        del speech_v2_test_client
        speech_v2_ok = True
    except DefaultCredentialsError as e:
        logger.error(f"[ConnectivityTest] FAILED SpeechClient (v2) creation due to DefaultCredentialsError: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"[ConnectivityTest] FAILED SpeechClient (v2) creation for test: {e}", exc_info=True)

    # Test Vertex AI Client
    try:
        logger.info(f"[ConnectivityTest] Attempting to initialize Vertex AI for project: {project_id}, location: {location}...")
        vertexai.init(project=project_id, location=location)
        logger.info("[ConnectivityTest] Vertex AI initialized successfully for test.")
        # Test Gemini model instantiation
        _ = GenerativeModel(asr_model_name_gemini)
        logger.info(f"[ConnectivityTest] Gemini model {asr_model_name_gemini} instantiated successfully for test.")
        vertex_ai_ok = True
    except DefaultCredentialsError as e:
        logger.error(f"[ConnectivityTest] FAILED Vertex AI init or Gemini model instantiation due to DefaultCredentialsError: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"[ConnectivityTest] FAILED Vertex AI init or Gemini model instantiation for test: {e}", exc_info=True)

    if not (speech_v2_ok and vertex_ai_ok):
        logger.critical("[ConnectivityTest] One or more Google Cloud connectivity tests FAILED. Check logs above.")
        return False
    logger.info("[ConnectivityTest] All Google Cloud connectivity tests PASSED.")
    return True


class TranscriptionServer:
    SAMPLING_RATE = 16000
    WINDOW_SIZE_SAMPLES = 1536 # This seems small for VAD window if SAMPLING_RATE is 16k (96ms)
    SPEECH_THRESHOLD = 0.5 # VAD threshold

    def __init__(self, project_id, location, recognizer_id_str): # recognizer_id_str is the raw string
        logger.info(f"Initializing TranscriptionServer with project='{project_id}', location='{location}', recognizer_id='{recognizer_id_str}'")
        torch.set_num_threads(1) # Good for CPU-bound torch operations if not using GPU

        self.PROJECT_ID = project_id
        self.LOCATION = location
        self.recognizer_id_str = recognizer_id_str # Store the raw ID

        # Perform connectivity tests before initializing clients for the class
        if not perform_google_cloud_connectivity_tests(self.PROJECT_ID, self.LOCATION):
            # Consider raising an error to halt server startup if connectivity is essential
            logger.error("Critical connectivity tests failed. TranscriptionServer may not function correctly.")
            # raise RuntimeError("Failed to establish initial connectivity to Google Cloud services.")

        # Initialize Google Cloud Clients ONCE
        self.speech_v2_client = None
        self.gemini_model_instance = None
        try:
            speech_endpoint = f"{self.LOCATION}-speech.googleapis.com"
            logger.info(f"Creating persistent SpeechClient (v2) with endpoint: {speech_endpoint}")
            self.speech_v2_client = SpeechClient(client_options=ClientOptions(api_endpoint=speech_endpoint))
            logger.info("Persistent SpeechClient (v2) created.")
        except Exception as e:
            logger.error(f"Failed to create persistent SpeechClient (v2): {e}", exc_info=True)
            # Server can continue but Chirp transcription will fail

        try:
            logger.info(f"Initializing persistent Vertex AI for project: {self.PROJECT_ID}, location: {self.LOCATION}")
            vertexai.init(project=self.PROJECT_ID, location=self.LOCATION) # Safe to call multiple times, but once is best
            logger.info("Persistent Vertex AI initialized.")
            logger.info(f"Creating persistent Gemini Model instance: {asr_model_name_gemini}")
            self.gemini_model_instance = GenerativeModel(asr_model_name_gemini)
            logger.info("Persistent Gemini Model instance created.")
        except Exception as e:
            logger.error(f"Failed to initialize persistent Vertex AI or Gemini Model: {e}", exc_info=True)
            # Server can continue but Gemini transcriptions will fail

        # VAD and local processing components
        self.vad_model = torch.jit.load('silero_vad/silero_vad.jit')
        self.vad_model_temp = torch.jit.load('silero_vad/silero_vad.jit') # Why a separate temp model?
        self.all_chunks = torch.tensor([])
        self.vad_speech_threshold = 0.7 # This overrides SPEECH_THRESHOLD from class level? Be consistent.
        self.min_silence_duration_ms = 100
        self.vad_iterator = VADIterator(model=self.vad_model, threshold=self.vad_speech_threshold,
                                        sampling_rate=self.SAMPLING_RATE,
                                        min_silence_duration_ms=self.min_silence_duration_ms)
        self.last_end = 0
        self.last_start = -1
        # self.all_transcriptions = [] # State per call or per instance? Seems per-call from usage
        # self.transcript_stream_results = [] # State per call or per instance?

    async def recv_audio_bytes(self, new_chunk, language_code):
        # Reset per-call state if necessary
        # self.all_transcriptions = []
        # self.transcript_stream_results = []
        # self.all_chunks = torch.tensor([]) # If all_chunks should be reset for each new full stream
        # self.last_start = -1
        # self.last_end = 0

        try:
            logger.info(f"recv_audio_bytes: Type={type(new_chunk)}, len={len(new_chunk)}, Lang={language_code}")
            audio_array = np.frombuffer(new_chunk, dtype=np.float32)
            current_chunks_tensor = torch.Tensor(audio_array)

            # process_new_chunks seems to manage its own state like self.all_chunks
            # This implies TranscriptionServer instance might be intended for a single continuous stream.
            # If new calls to recv_audio_bytes are for *different* streams, state needs careful management.
            result = await self.process_new_chunks(current_chunks_tensor, language_code)
            return result
        except Exception as e:
            logger.error(f"Error in recv_audio_bytes: {e}", exc_info=True)
            return None # Or an error protobuf message

    def recv_audio_output(self, current_transcript_segments):
        # ... (your existing implementation seems okay, uses stt__pb2) ...
        if current_transcript_segments and len(current_transcript_segments) > 0:
            # (Your existing logic)
            transcript_stream_results_list = []
            first_start_offset = current_transcript_segments[0].get('start', 0)

            for segment in current_transcript_segments:
                # Ensure all required keys exist or have defaults
                result_end_offset_val = segment.get('end', segment.get('immediate', 0))
                is_final_val = 'end' in segment
                transcript_val = segment.get('transcript', "")
                translation_val = segment.get('translation', "") # Will be None if not translated
                confidence_val = segment.get('confidence', 0.2) # Default confidence

                alternatives_list = [stt__pb2.Alternative(
                    transcript=transcript_val,
                    translation=translation_val,
                    confidence=confidence_val
                    # synthesized_speech_audio field is populated by stt_server.py's Listener
                )]
                
                transcript_stream_results_list.append(stt__pb2.TranscriptStreamResult(
                    result_end_offset=int(result_end_offset_val), # Ensure it's int
                    is_final=is_final_val,
                    alternatives=alternatives_list
                ))
            return stt__pb2.TranscriptStreamResponse(
                speech_event_offset=int(first_start_offset), # Ensure it's int
                results=transcript_stream_results_list
            )
        else:
            logger.warning("recv_audio_output called with no segments.")
            return None
    
    async def process_new_chunks(self, current_chunks_tensor, language_code):
        # ... (your existing VAD logic using self.all_chunks, self.vad_iterator etc.) ...
        # This function is quite long and stateful (self.all_chunks, self.last_start).
        # The core change needed is within the calls to self.transcribe and self.transcribe_by_gemini.

        # --- Inside the loop where you call transcription ---
        # ...
        # if segment needs transcription:
        #    torch_chunks = current_all_chunks[current_start_index : current_end_index]
        #    transcripted_base64_content = self.tensor_to_base64(torch_chunks, self.SAMPLING_RATE)
        #
        #    if some_condition_to_use_chirp:
        #        transcript_json_result = await self.transcribe(transcripted_base64_content, language_code)
        #    else: # use Gemini
        #        transcript_json_result = await self.transcribe_by_gemini(transcripted_base64_content, LANGUAGE_CODE_DIC[language_code])
        #
        #    if 'end' in segment: # Assuming translation only for final segments
        #        translation_json_result = await self.transcribe_and_translate_by_gemini(
        #            transcripted_base64_content,
        #            LANGUAGE_CODE_DIC[language_code],
        #            TARGET_LANGUAGE
        #        )
        # ...
        # For brevity, I'm not reproducing the entire process_new_chunks logic,
        # focus is on how transcribe/transcribe_by_gemini are called from it.
        # Ensure these calls are made correctly after the VAD logic.
        # The dummy return from my previous example needs to be replaced by your actual VAD + transcription calls.
        logger.warning("process_new_chunks: Placeholder logic - returning dummy response. Implement VAD and transcription calls.")
        # This is just to make the server runnable for testing the gRPC layer itself.
        # You need to integrate your VAD logic here.
        if len(self.all_chunks) > self.SAMPLING_RATE * 1: # if more than 1 sec of audio
             dummy_segments = [{
                'start': 0, 'end': len(self.all_chunks),
                'transcript': f"Processed dummy for {language_code}",
                'translation': "Dummy translation"
            }]
             self.all_chunks = torch.tensor([]) # Reset for next time
             return self.recv_audio_output(dummy_segments)
        return None # No output if not enough audio

    def tensor_to_base64(self, tensor, sample_rate):
        # ... (your existing implementation - seems okay) ...
        audio_buffer = io.BytesIO()
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        torchaudio.save(audio_buffer, tensor, sample_rate, format="wav", bits_per_sample=16) # Ensure format="wav"
        audio_bytes = audio_buffer.getvalue()
        base64_bytes = base64.b64encode(audio_bytes)
        base64_string = base64_bytes.decode("utf-8")
        return base64_string

    async def transcribe(self, base64_data_wav, language_code: str): # Renamed for clarity
        if not self.speech_v2_client:
            logger.error("Chirp STT: SpeechClient (v2) not initialized. Cannot transcribe.")
            return ""

        start_time_ms = int(datetime.now().timestamp() * 1000)
        logger.info(f"Chirp STT: Transcribing for lang '{language_code}'. Data length (b64): {len(base64_data_wav)}")

        recognition_config = cloud_speech.RecognitionConfig(
            explicit_decoding_config=cloud_speech.ExplicitDecodingConfig(
                encoding=cloud_speech.ExplicitDecodingConfig.AudioEncoding.LINEAR16, # WAV from torchaudio is LINEAR16
                sample_rate_hertz=self.SAMPLING_RATE,
                audio_channel_count=1
            ),
            language_codes=[language_code],
            model="chirp" # General Chirp model
        )

        request_params = {
            "config": recognition_config,
            "content": base64.b64decode(base64_data_wav), # Decode base64 to bytes
        }

        # Only add the 'recognizer' field if a specific recognizer ID (not "_" or empty) is provided
        if self.recognizer_id_str and self.recognizer_id_str not in ["_", ""]:
            recognizer_path = f"projects/{self.PROJECT_ID}/locations/{self.LOCATION}/recognizers/{self.recognizer_id_str}"
            request_params["recognizer"] = recognizer_path
            logger.info(f"Chirp STT: Using specific recognizer: {recognizer_path}")
        else:
            logger.info("Chirp STT: Using general model 'chirp' without a specific recognizer resource.")

        request = cloud_speech.RecognizeRequest(**request_params)
        transcript_text = ""

        try:
            loop = asyncio.get_running_loop()
            logger.debug("Chirp STT: Calling client.recognize via run_in_executor...")
            # client.recognize is synchronous, run in executor
            response = await loop.run_in_executor(None, functools.partial(self.speech_v2_client.recognize, request=request))
            
            end_time_ms = int(datetime.now().timestamp() * 1000)
            duration_ms = end_time_ms - start_time_ms
            logger.info(f"Chirp STT: Response received in {duration_ms} ms.")

            if response.results and response.results[0].alternatives:
                transcript_text = response.results[0].alternatives[0].transcript
                billed_duration_s = response.metadata.total_billed_duration.seconds if response.metadata and response.metadata.total_billed_duration else "N/A"
                logger.info(f"Chirp STT: Transcript='{transcript_text}', Billed Duration='{billed_duration_s}s'")
                transcript_text = self.process_ununsed(transcript_text) # Your processing function
            else:
                logger.warning("Chirp STT: No transcript returned in response.")
        except grpc.RpcError as e: # google.api_core.exceptions.GoogleAPIError might also be relevant
            logger.error(f"Chirp STT: gRPC error during recognize call: Code={e.code()} Details='{e.details()}'", exc_info=True)
        except Exception as e:
            logger.error(f"Chirp STT: Generic error during recognize call: {e}", exc_info=True)
        
        return transcript_text

    async def call_gemini(self, prompt_contents, generation_config, safety_settings, model_instance):
        if not model_instance:
             logger.error("Gemini Call: Model instance is None. Cannot proceed.")
             return None # Or raise error
        try:
            loop = asyncio.get_running_loop()
            logger.debug("Gemini Call: Calling model.generate_content via run_in_executor...")
            # model_instance.generate_content is synchronous
            response = await loop.run_in_executor(
                None,
                functools.partial(
                    model_instance.generate_content, # The method to call
                    prompt_contents,                 # First positional argument to generate_content
                    generation_config=generation_config,
                    safety_settings=safety_settings,
                    stream=False
                )
            )
            logger.debug("Gemini Call: Response received.")
            return response
        except Exception as e:
            logger.error(f"Gemini Call: Error during generate_content: {e}", exc_info=True)
            return None # Or raise error

    async def transcribe_by_gemini(self, audio_base64_wav, language_name): # language_name e.g. "English"
        if not self.gemini_model_instance:
            logger.error("Gemini ASR: Gemini model not initialized. Cannot transcribe.")
            return ""
        
        logger.info(f"Gemini ASR: Transcribing for lang '{language_name}'. Data length (b64): {len(audio_base64_wav)}")
        start_time_ms = int(datetime.now().timestamp() * 1000)

        generation_config = {"max_output_tokens": 256, "temperature": 0.1, "top_p": 0.95, "response_mime_type": "application/json"}
        safety_settings = {category: generative_models.HarmBlockThreshold.BLOCK_NONE for category in generative_models.HarmCategory}
        
        from prompts import prompt_template_asr # Ensure this is accessible
        prompt = prompt_template_asr.format(language=language_name)
        prompt_contents = [prompt, Part.from_data(mime_type="audio/wav", data=base64.b64decode(audio_base64_wav))]
        
        transcript = ""
        response = await self.call_gemini(prompt_contents, generation_config, safety_settings, self.gemini_model_instance)

        if response and hasattr(response, 'text'):
            try:
                response_results = json.loads(response.text)
                transcript = response_results.get('Fluent_Transcription', "")
            except json.JSONDecodeError:
                logger.error(f"Gemini ASR: Failed to parse JSON from response: {response.text}")
            except Exception as e:
                logger.error(f"Gemini ASR: Error processing Gemini response: {e}", exc_info=True)
        else:
            logger.warning("Gemini ASR: No valid response received from Gemini model.")

        end_time_ms = int(datetime.now().timestamp() * 1000)
        duration_ms = end_time_ms - start_time_ms
        logger.info(f"Gemini ASR: Transcript='{transcript}', Duration={duration_ms}ms")
        return self.process_ununsed(transcript)

    async def transcribe_and_translate_by_gemini(self, audio_base64_wav, source_language_name, target_language_name):
        if not self.gemini_model_instance:
            logger.error("Gemini AST: Gemini model not initialized. Cannot process.")
            return ""

        logger.info(f"Gemini AST: Translating from '{source_language_name}' to '{target_language_name}'. Data length (b64): {len(audio_base64_wav)}")
        start_time_ms = int(datetime.now().timestamp() * 1000)

        generation_config = {"max_output_tokens": 256, "temperature": 0.1, "top_p": 0.95, "response_mime_type": "application/json"}
        safety_settings = {category: generative_models.HarmBlockThreshold.BLOCK_NONE for category in generative_models.HarmCategory}

        from prompts import prompt_template_ast # Ensure this is accessible
        prompt = prompt_template_ast.format(source_language=source_language_name, target_language=target_language_name)
        prompt_contents = [prompt, Part.from_data(mime_type="audio/wav", data=base64.b64decode(audio_base64_wav))]
        
        translation = ""
        response = await self.call_gemini(prompt_contents, generation_config, safety_settings, self.gemini_model_instance)

        if response and hasattr(response, 'text'):
            try:
                response_results = json.loads(response.text)
                translation = response_results.get('Translation', "")
            except json.JSONDecodeError:
                logger.error(f"Gemini AST: Failed to parse JSON from response: {response.text}")
            except Exception as e:
                logger.error(f"Gemini AST: Error processing Gemini response: {e}", exc_info=True)
        else:
            logger.warning("Gemini AST: No valid response received from Gemini model.")
            
        end_time_ms = int(datetime.now().timestamp() * 1000)
        duration_ms = end_time_ms - start_time_ms
        logger.info(f"Gemini AST: Translation='{translation}', Duration={duration_ms}ms")
        return self.process_ununsed(translation)

    def process_ununsed(self, txt): # Should this be static or self is not used?
        # ... (your existing implementation) ...
        if not isinstance(txt, str): # Add a type check for safety
            logger.warning(f"process_ununsed expected string, got {type(txt)}. Returning empty string.")
            return ""
        txt = txt.replace("\n", "").replace("`", "").replace("삐", "")
        txt = txt.replace("<spacing>","").replace("<noise>","").replace("<spoken_noise>", "")
        return txt.lower().replace("null", "").strip()

    # save_tensor_to_wav - seems fine
    # find_first_no_transcript_segment - seems fine, though "StreamingRecognizeResponse" line is just a comment.
    # cleanup - seems fine