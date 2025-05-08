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
import grpc # Added for grpc.RpcError
import os

from google.api_core.client_options import ClientOptions
from google.cloud.speech_v2 import SpeechClient
from google.cloud.speech_v2.types import cloud_speech
from google.auth.exceptions import DefaultCredentialsError

import vertexai
from vertexai.generative_models import GenerativeModel, Part
import vertexai.generative_models as generative_models

import stt_pb2 as stt__pb2
from vad import VADIterator
# from utils_vad import get_speech_timestamps

from prompts import prompt_template_ast, prompt_template_asr

asr_model_name_gemini = "gemini-2.0-flash-lite-001"
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

def perform_google_cloud_connectivity_tests(project_id, location):
    # ... (implementation as you provided) ...
    logger.info("[ConnectivityTest] Starting Google Cloud connectivity tests...")
    speech_v2_ok = False
    vertex_ai_ok = False
    try:
        endpoint = f"{location}-speech.googleapis.com"
        logger.info(f"[ConnectivityTest] Attempting to create SpeechClient (v2) with endpoint: {endpoint}...")
        speech_v2_test_client = SpeechClient(client_options=ClientOptions(api_endpoint=endpoint))
        logger.info("[ConnectivityTest] SpeechClient (v2) created successfully for test.")
        del speech_v2_test_client
        speech_v2_ok = True
    except DefaultCredentialsError as e:
        logger.error(f"[ConnectivityTest] FAILED SpeechClient (v2) creation due to DefaultCredentialsError: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"[ConnectivityTest] FAILED SpeechClient (v2) creation for test: {e}", exc_info=True)
    try:
        logger.info(f"[ConnectivityTest] Attempting to initialize Vertex AI for project: {project_id}, location: {location}...")
        vertexai.init(project=project_id, location=location)
        logger.info("[ConnectivityTest] Vertex AI initialized successfully for test.")
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
    WINDOW_SIZE_SAMPLES = 1024 # Your current setting (64ms)
    SPEECH_THRESHOLD = 0.33    # Your current setting

    def __init__(self, project_id, location, recognizer_id_str):
        # ... (implementation as you provided, including client initializations) ...
        logger.info(f"Initializing TranscriptionServer with project='{project_id}', location='{location}', recognizer_id='{recognizer_id_str}'")
        torch.set_num_threads(1)
        self.PROJECT_ID = project_id
        self.LOCATION = location
        self.recognizer_id_str = recognizer_id_str
        if not perform_google_cloud_connectivity_tests(self.PROJECT_ID, self.LOCATION):
            logger.error("Critical connectivity tests failed. TranscriptionServer may not function correctly.")
        self.speech_v2_client = None
        self.gemini_model_instance = None
        try:
            speech_endpoint = f"{self.LOCATION}-speech.googleapis.com"
            logger.info(f"Creating persistent SpeechClient (v2) with endpoint: {speech_endpoint}")
            self.speech_v2_client = SpeechClient(client_options=ClientOptions(api_endpoint=speech_endpoint))
            logger.info("Persistent SpeechClient (v2) created.")
        except Exception as e:
            logger.error(f"Failed to create persistent SpeechClient (v2): {e}", exc_info=True)
        try:
            logger.info(f"Initializing persistent Vertex AI for project: {self.PROJECT_ID}, location: {self.LOCATION}")
            vertexai.init(project=self.PROJECT_ID, location=self.LOCATION)
            logger.info("Persistent Vertex AI initialized.")
            logger.info(f"Creating persistent Gemini Model instance: {asr_model_name_gemini}")
            self.gemini_model_instance = GenerativeModel(asr_model_name_gemini)
            logger.info("Persistent Gemini Model instance created.")
        except Exception as e:
            logger.error(f"Failed to initialize persistent Vertex AI or Gemini Model: {e}", exc_info=True)
        self.vad_model = torch.jit.load('silero_vad/silero_vad.jit')
        self.all_chunks = torch.tensor([])
        self.vad_speech_threshold_iterator = 0.5
        self.min_silence_duration_ms = 400 # Your current setting
        self.vad_iterator = VADIterator(model=self.vad_model, threshold=self.vad_speech_threshold_iterator,
                                        sampling_rate=self.SAMPLING_RATE,
                                        min_silence_duration_ms=self.min_silence_duration_ms)
        self.last_end = 0
        self.last_start = -1


    async def recv_audio_bytes(self, new_chunk, language_code):
        # ... (implementation as you provided) ...
        try:
            logger.info(f"Type={type(new_chunk)}, len={len(new_chunk)}, Lang={language_code}")
            audio_array = np.frombuffer(new_chunk, dtype=np.float32)
            current_chunks_tensor = torch.from_numpy(audio_array.copy())
            if current_chunks_tensor.numel() > 0: 
                max_val = torch.abs(current_chunks_tensor).max()
                if max_val > 0: 
                    current_chunks_tensor = current_chunks_tensor / max_val
                    logger.debug(f"Normalized incoming browser audio chunk. Original max_val: {max_val:.4f}")
            result = await self.process_new_chunks(current_chunks_tensor, language_code)
            return result
        except Exception as e:
            logger.error(f"Error in recv_audio_bytes: {e}", exc_info=True)
            return None

    def recv_audio_output(self, current_transcript_segments):
        # ... (implementation as you provided, ensuring it uses stt_duration_ms and translation_duration_ms) ...
        if current_transcript_segments and len(current_transcript_segments) > 0:
            transcript_stream_results_list = []
            first_start_offset = current_transcript_segments[0].get('start', 0)
            for segment in current_transcript_segments:
                result_end_offset_val = segment.get('end', segment.get('immediate', 0))
                is_final_val = 'end' in segment
                transcript_val = segment.get('transcript', "")
                translation_val = segment.get('translation', "")
                confidence_val = segment.get('confidence', 0.2) 
                stt_duration_val = segment.get('stt_duration', 0) 
                translation_duration_val = segment.get('translation_duration', 0)
                alternatives_list = [stt__pb2.Alternative(
                    transcript=transcript_val,
                    translation=translation_val,
                    confidence=confidence_val,
                    stt_duration_ms=stt_duration_val,
                    translation_duration_ms=translation_duration_val
                )]
                transcript_stream_results_list.append(stt__pb2.TranscriptStreamResult(
                    result_end_offset=int(result_end_offset_val),
                    is_final=is_final_val,
                    alternatives=alternatives_list
                ))
            return stt__pb2.TranscriptStreamResponse(
                speech_event_offset=int(first_start_offset),
                results=transcript_stream_results_list
            )
        else:
            logger.debug("recv_audio_output called with no segments.")
            return None

    # +++ NEW HELPER METHOD for concurrent processing of one segment +++
    async def _process_single_segment_concurrently(self, segment_dict, audio_b64_content, language_code_for_stt, target_gemini_language_for_ast):
        """
        Processes a single audio segment for STT and (conditionally) Translation.
        Modifies segment_dict in place and returns it.
        """
        current_start_index = segment_dict['start'] # For logging context
        logger.info(f"Async task started for segment: start={current_start_index}")

        segment_dict['stt_duration'] = 0
        segment_dict['translation_duration'] = 0
        segment_dict['transcript'] = ""
        segment_dict['translation'] = ""

        try:
            # Perform transcription
            transcript_text, stt_duration = await self.transcribe_by_gemini(audio_b64_content, target_gemini_language_for_ast)
            segment_dict['transcript'] = transcript_text
            segment_dict['stt_duration'] = stt_duration
            logger.info(f"Segment transcript (async): '{transcript_text}' (STT: {stt_duration}ms) for start={current_start_index}")

            # Perform translation if the segment is final and transcription was successful
            if 'end' in segment_dict and transcript_text:
                logger.info(f"Segment (async) is final, attempting translation. Source: {target_gemini_language_for_ast}, Target: {TARGET_LANGUAGE} for start={current_start_index}")
                if target_gemini_language_for_ast != TARGET_LANGUAGE:
                    translation_text, translation_duration = await self.transcribe_and_translate_by_gemini(
                        audio_b64_content, # Using same audio for translation task
                        target_gemini_language_for_ast,
                        TARGET_LANGUAGE
                    )
                    segment_dict['translation'] = translation_text
                    segment_dict['translation_duration'] = translation_duration
                    logger.info(f"Segment translation (async): '{translation_text}' (Translate: {translation_duration}ms) for start={current_start_index}")
                else:
                    segment_dict['translation'] = transcript_text # Same language, use transcript
                    segment_dict['translation_duration'] = 0
                    logger.info(f"Source and target language same for translation ('{TARGET_LANGUAGE}') for start={current_start_index}.")
            elif 'end' in segment_dict and not transcript_text:
                logger.warning(f"Segment (async) is final but no transcript to translate for start={current_start_index}")

        except Exception as e:
            logger.error(f"Error processing segment (async) start={current_start_index}: {e}", exc_info=True)
            # Ensure basic fields are still present even on error
            segment_dict['transcript'] = segment_dict.get('transcript', "[STT_ERROR]")
            segment_dict['translation'] = segment_dict.get('translation', "[TRANSLATE_ERROR]")
        
        return segment_dict # Return the modified dictionary


    async def process_new_chunks(self, current_chunks, language_code):
        # --- Start of your existing VAD and segment identification logic ---
        # (This part remains largely the same as your provided version until
        #  you get the `current_transcript_segments_to_process` list)
        last_round_end = ((int)(len(self.all_chunks)/self.WINDOW_SIZE_SAMPLES))*self.WINDOW_SIZE_SAMPLES
        current_last_start = self.last_start
        current_all_segments = [] 
        self.all_chunks = torch.cat([self.all_chunks, current_chunks])
        current_all_chunks_for_vad = self.all_chunks.clone()
        has_new_speech = False
        for i in range(last_round_end, len(current_all_chunks_for_vad), self.WINDOW_SIZE_SAMPLES):
            loop_end_index = i + self.WINDOW_SIZE_SAMPLES
            chunk = current_all_chunks_for_vad[i: loop_end_index]
            if loop_end_index > last_round_end and len(chunk) == self.WINDOW_SIZE_SAMPLES:
                if self.vad_model(chunk, self.SAMPLING_RATE).item() > self.SPEECH_THRESHOLD:
                    has_new_speech = True
            if loop_end_index >= len(current_all_chunks_for_vad) -1: 
                logging.debug("End of the audio chunk detected in VAD loop.")
                if current_last_start != -1: 
                    current_all_segments.append({'start': current_last_start, 'immediate': i + len(chunk)})
                break
            speech_dict = self.vad_iterator(chunk, return_seconds=False)
            if speech_dict: 
                if 'end' in speech_dict:
                    logging.info(f"VADIterator found 'end': last_start={current_last_start}, end_offset={speech_dict['end']}")
                    if current_last_start != -1:
                         current_all_segments.append({'start': current_last_start, 'end': speech_dict['end']})
                    else: 
                         logger.warning(f"VADIterator found 'end' but current_last_start was -1. Segment: {speech_dict}")
                    current_last_start = -1 
                    self.last_start = -1 
                elif 'start' in speech_dict:
                    current_last_start = speech_dict['start']
                    self.last_start = speech_dict['start'] 
                    logging.info(f"VADIterator found 'start': start_offset={current_last_start}")
        logger.info(f"VAD processing complete. has_new_speech flag: {has_new_speech}, detected segments: {len(current_all_segments)}")
        if not current_all_segments and not has_new_speech:
            logger.debug("No new speech segments detected by VAD.")
            return None
        # --- Your existing save_tensor_to_wav for "to_website_..." can remain here if desired ---
        # ... (your filtering logic to create `valid_segments_for_transcription`) ...
        valid_segments_for_transcription = []
        temp_start_for_filtering = 0 
        if current_all_segments:
             temp_start_for_filtering = current_all_segments[0].get('start',0)
        for idx, seg in enumerate(current_all_segments):
            is_last_segment = (idx == len(current_all_segments) - 1)
            if 'end' in seg:
                seg_start = seg.get('start', temp_start_for_filtering) 
                if seg_start < seg['end']: 
                    valid_segments_for_transcription.append({'start': seg_start, 'end': seg['end']})
                    temp_start_for_filtering = seg['end']
            elif 'immediate' in seg and is_last_segment:
                seg_start = seg.get('start', temp_start_for_filtering)
                if seg_start < seg['immediate'] and (seg['immediate'] - seg_start > self.SAMPLING_RATE * 0.4): 
                    valid_segments_for_transcription.append({'start': seg_start, 'immediate': seg['immediate']})
        logger.info(f"Filtered valid_segments_for_transcription: {valid_segments_for_transcription}")
        if not valid_segments_for_transcription:
            logger.debug("No valid segments identified for transcription after filtering.")
            return None
        # ... (your consolidation logic for `valid_segments_for_transcription`) ...
        if len(valid_segments_for_transcription) > 2:
            logger.info("Consolidating segments for transcription.")
            consolidated_end = valid_segments_for_transcription[-2]['end'] if 'end' in valid_segments_for_transcription[-2] else valid_segments_for_transcription[-2]['immediate']
            valid_segments_for_transcription = [
                {'start': valid_segments_for_transcription[0]['start'], 'end': consolidated_end},
                valid_segments_for_transcription[-1] 
            ]
            logger.info(f"Consolidated segments: {valid_segments_for_transcription}")
        # --- End of VAD and segment identification logic ---
        # `valid_segments_for_transcription` is now the list of segments to process concurrently

        if not valid_segments_for_transcription:
            logger.debug("No segments to submit for ASR after VAD processing.")
            return None

        asr_processing_tasks = []
        
        for segment_info_from_vad in valid_segments_for_transcription:
            current_start_index = segment_info_from_vad['start']
            current_end_index = segment_info_from_vad.get('end', segment_info_from_vad.get('immediate'))

            if current_end_index is None or current_end_index <= current_start_index:
                logger.warning(f"Skipping invalid segment for ASR task creation: start={current_start_index}, end={current_end_index}")
                # To maintain order with asyncio.gather, we might need to add a placeholder or handle this.
                # For now, we'll skip creating a task for it, which means it won't be in the output if invalid.
                # A more robust solution would filter these out *before* this loop.
                continue

            logger.info(f"Preparing ASR task for segment: start={current_start_index}, end={current_end_index}")
            torch_segment_chunks = self.all_chunks[current_start_index:current_end_index]
            
            if torch_segment_chunks.numel() == 0:
                logger.warning(f"Segment resulted in empty torch_chunks: start={current_start_index}, end={current_end_index}. Creating dummy processed segment.")
                # If a segment is empty, create a task that resolves immediately with empty results
                # Or, handle this inside _process_single_segment_concurrently
                async def empty_segment_handler(seg_dict):
                    seg_dict['transcript'] = ""
                    seg_dict['stt_duration'] = 0
                    seg_dict['translation'] = ""
                    seg_dict['translation_duration'] = 0
                    return seg_dict
                task = asyncio.create_task(empty_segment_handler(segment_info_from_vad.copy()))
                asr_processing_tasks.append(task)
                continue # Skip saving and base64 encoding for empty chunk

            # Save debug audio (your existing logic)
            try:
                debug_audio_dir = "debug_audio_clips"
                if not os.path.exists(debug_audio_dir): os.makedirs(debug_audio_dir)
                timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                filename = os.path.join(debug_audio_dir, f"to_gemini_{timestamp_str}_s{current_start_index}_e{current_end_index}.wav")
                self.save_tensor_to_wav(torch_segment_chunks, self.SAMPLING_RATE, filename)
                logger.info(f"Saved debug audio segment for Gemini to: {filename}")
            except Exception as save_e:
                logger.error(f"Failed to save debug audio segment: {save_e}", exc_info=True)

            transcripted_base64_content = self.tensor_to_base64(torch_segment_chunks, self.SAMPLING_RATE)
            target_gemini_language_for_ast = LANGUAGE_CODE_DIC.get(language_code, "English")
            
            # Pass a copy of segment_info_from_vad to avoid race conditions if dicts were complexly shared (less of an issue here as it's modified in place by the task)
            task = asyncio.create_task(
                self._process_single_segment_concurrently(
                    segment_info_from_vad.copy(), # Pass a copy of the dict
                    transcripted_base64_content,
                    language_code, # Original language_code from client for STT config if Chirp was used
                    target_gemini_language_for_ast # Mapped language name for Gemini prompts
                )
            )
            asr_processing_tasks.append(task)

        if not asr_processing_tasks:
            logger.debug("No ASR tasks were created.")
            return None
            
        logger.info(f"Awaiting completion of {len(asr_processing_tasks)} concurrent ASR/AST tasks...")
        # Results will be a list of the modified segment dictionaries, in the same order as tasks.
        output_segments_with_results = await asyncio.gather(*asr_processing_tasks)
        logger.info(f"All {len(asr_processing_tasks)} ASR/AST tasks completed.")

        # Filter out any potential None results if _process_single_segment_concurrently could return None on error
        output_segments_with_results = [res for res in output_segments_with_results if res is not None]

        if not output_segments_with_results:
            logger.debug("No segments produced results after concurrent ASR/AST processing.")
            return None
            
        return self.recv_audio_output(output_segments_with_results)

    # ... (tensor_to_base64, transcribe, call_gemini, transcribe_by_gemini, transcribe_and_translate_by_gemini methods remain as you provided,
    #      ensure `transcribe_by_gemini` and `transcribe_and_translate_by_gemini` return (text, duration_ms) tuples.
    #      Your provided code for these already does that, so that's good.)

    # ... (process_ununsed, cleanup, save_tensor_to_wav, find_first_no_transcript_segment methods as you provided) ...
    # Make sure the transcribe (Chirp) method also returns (text, duration_ms) if it's ever used.
    async def transcribe(self, base64_data_wav, language_code: str):
        if not self.speech_v2_client:
            logger.error("Chirp STT: SpeechClient (v2) not initialized. Cannot transcribe.")
            return "", 0 # Return text and duration
        # ... (rest of your transcribe method, ensure it calculates and returns duration_ms)
        # Example from your code (needs start_time_ms, end_time_ms defined correctly)
        start_time_ms = int(datetime.now().timestamp() * 1000)
        # ... your existing Chirp client.recognize call ...
        # response = await loop.run_in_executor(...)
        end_time_ms = int(datetime.now().timestamp() * 1000)
        duration_ms = end_time_ms - start_time_ms
        # ... (process response to get transcript_text) ...
        # return self.process_ununsed(transcript_text), duration_ms
        # Using the full implementation from your provided code:
        logger.info(f"Chirp STT: Transcribing for lang '{language_code}'. Data length (b64): {len(base64_data_wav)}")
        recognition_config = cloud_speech.RecognitionConfig(
            explicit_decoding_config=cloud_speech.ExplicitDecodingConfig(
                encoding=cloud_speech.ExplicitDecodingConfig.AudioEncoding.LINEAR16,
                sample_rate_hertz=self.SAMPLING_RATE,
                audio_channel_count=1
            ),
            language_codes=[language_code], model="chirp"
        )
        request_params = {"config": recognition_config, "content": base64.b64decode(base64_data_wav)}
        if self.recognizer_id_str and self.recognizer_id_str not in ["_", ""]:
            recognizer_path = f"projects/{self.PROJECT_ID}/locations/{self.LOCATION}/recognizers/{self.recognizer_id_str}"
            request_params["recognizer"] = recognizer_path
            logger.info(f"Chirp STT: Using specific recognizer: {recognizer_path}")
        else:
            logger.info("Chirp STT: Using general model 'chirp' without a specific recognizer resource.")
        request = cloud_speech.RecognizeRequest(**request_params)
        transcript_text = ""
        actual_duration_ms = 0
        try:
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(None, functools.partial(self.speech_v2_client.recognize, request=request))
            end_time_ms_call = int(datetime.now().timestamp() * 1000)
            actual_duration_ms = end_time_ms_call - start_time_ms # Renamed to avoid conflict if already defined
            logger.info(f"Chirp STT: Response received in {actual_duration_ms} ms.")
            if response.results and response.results[0].alternatives:
                transcript_text = response.results[0].alternatives[0].transcript
                billed_duration_s = response.metadata.total_billed_duration.seconds if response.metadata and response.metadata.total_billed_duration else "N/A"
                logger.info(f"Chirp STT: Transcript='{transcript_text}', Billed Duration='{billed_duration_s}s'")
                transcript_text = self.process_ununsed(transcript_text)
            else:
                logger.warning("Chirp STT: No transcript returned in response.")
        except grpc.RpcError as e:
            logger.error(f"Chirp STT: gRPC error during recognize call: Code={e.code()} Details='{e.details()}'", exc_info=True)
        except Exception as e:
            logger.error(f"Chirp STT: Generic error during recognize call: {e}", exc_info=True)
        return transcript_text, actual_duration_ms


    async def call_gemini(self, prompt_contents, generation_config, safety_settings, model_instance):
        # ... (implementation as you provided) ...
        if not model_instance:
             logger.error("Gemini Call: Model instance is None. Cannot proceed.")
             return None
        try:
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(
                None, functools.partial(
                    model_instance.generate_content, prompt_contents,
                    generation_config=generation_config, safety_settings=safety_settings, stream=False
                )
            )
            return response
        except Exception as e:
            logger.error(f"Gemini Call: Error during generate_content: {e}", exc_info=True)
            return None

    async def transcribe_by_gemini(self, audio_base64_wav, language_name):
        # ... (implementation as you provided, ensure it returns (text, duration_ms)) ...
        if not self.gemini_model_instance:
            logger.error("Gemini ASR: Gemini model not initialized. Cannot transcribe.")
            return "", 0 
        logger.info(f"Gemini ASR: Transcribing for lang '{language_name}'. Data length (b64): {len(audio_base64_wav)}")
        start_time_ms = int(datetime.now().timestamp() * 1000)
        generation_config = {"max_output_tokens": 512, "temperature": 0.1, "top_p": 0.95, "response_mime_type": "application/json"}
        safety_settings = {category: generative_models.HarmBlockThreshold.BLOCK_NONE for category in generative_models.HarmCategory}
        prompt = prompt_template_asr.format(language=language_name)
        prompt_contents = [prompt, Part.from_data(mime_type="audio/wav", data=base64.b64decode(audio_base64_wav))]
        transcript = ""
        response = await self.call_gemini(prompt_contents, generation_config, safety_settings, self.gemini_model_instance)
        # Check if response is None (from error in call_gemini)
        if response is None:
            logger.warning("Gemini ASR: call_gemini returned None, possibly due to an error.")
        elif hasattr(response, 'text'):
            logger.info(f"Gemini Call transcribe: Response received: {response.text[:200]}...") # Log snippet
            try:
                response_results = json.loads(response.text)
                transcript = response_results.get('Fluent_Transcription', "")
            except json.JSONDecodeError:
                logger.error(f"Gemini ASR: Failed to parse JSON from response: {response.text}")
            except Exception as e:
                logger.error(f"Gemini ASR: Error processing Gemini response: {e}", exc_info=True)
        else:
            logger.warning(f"Gemini ASR: No valid 'text' attribute in response received from Gemini model. Response: {type(response)}")
        end_time_ms = int(datetime.now().timestamp() * 1000)
        duration_ms = end_time_ms - start_time_ms
        logger.info(f"Gemini ASR: Transcript='{transcript}', Duration={duration_ms}ms")
        return self.process_ununsed(transcript), duration_ms

    async def transcribe_and_translate_by_gemini(self, audio_base64_wav, source_language_name, target_language_name):
        # ... (implementation as you provided, ensure it returns (text, duration_ms)) ...
        if not self.gemini_model_instance:
            logger.error("Gemini AST: Gemini model not initialized. Cannot process.")
            return "", 0
        logger.info(f"Gemini AST: Translating from '{source_language_name}' to '{target_language_name}'. Data length (b64): {len(audio_base64_wav)}")
        start_time_ms = int(datetime.now().timestamp() * 1000)
        generation_config = {"max_output_tokens": 512, "temperature": 0.1, "top_p": 0.95, "response_mime_type": "application/json"}
        safety_settings = {category: generative_models.HarmBlockThreshold.BLOCK_NONE for category in generative_models.HarmCategory}
        prompt = prompt_template_ast.format(source_language=source_language_name, target_language=target_language_name)
        prompt_contents = [prompt, Part.from_data(mime_type="audio/wav", data=base64.b64decode(audio_base64_wav))]
        translation = ""
        response = await self.call_gemini(prompt_contents, generation_config, safety_settings, self.gemini_model_instance)
        if response is None:
             logger.warning("Gemini AST: call_gemini returned None, possibly due to an error.")
        elif hasattr(response, 'text'):
            logger.info(f"Gemini Call translate: Response received: {response.text[:200]}...") # Log snippet
            try:
                response_results = json.loads(response.text)
                translation = response_results.get('Translation', "")
            except json.JSONDecodeError:
                logger.error(f"Gemini AST: Failed to parse JSON from response: {response.text}")
            except Exception as e:
                logger.error(f"Gemini AST: Error processing Gemini response: {e}", exc_info=True)
        else:
            logger.warning(f"Gemini AST: No valid 'text' attribute in response received from Gemini model. Response: {type(response)}")
        end_time_ms = int(datetime.now().timestamp() * 1000)
        duration_ms = end_time_ms - start_time_ms
        logger.info(f"Gemini AST: Translation='{translation}', Duration={duration_ms}ms")
        return self.process_ununsed(translation), duration_ms

    # ... (process_ununsed, cleanup, save_tensor_to_wav, find_first_no_transcript_segment as you provided)
    # Ensure find_first_no_transcript_segment removes the stray "StreamingRecognizeResponse" line
    def find_first_no_transcript_segment(self, segments):
        for segment in segments:
            if not 'transcript' in segment:
                return segment
        return None

    def process_ununsed(self, txt):
        if not isinstance(txt, str):
            logger.warning(f"process_ununsed expected string, got {type(txt)}. Returning empty string.")
            return ""
        txt = txt.replace("\n", "").replace("`", "").replace("삐", "")
        txt = txt.replace("<spacing>","").replace("<noise>","").replace("<spoken_noise>", "")
        return txt.lower().replace("null", "").strip()
    
    def cleanup(self):
        logger.info("Cleaning up TranscriptionServer state for a new stream.")
        self.all_chunks = torch.tensor([])
        self.last_end = 0
        self.last_start = -1
        if hasattr(self.vad_iterator, 'reset_states') and callable(self.vad_iterator.reset_states):
            self.vad_iterator.reset_states()

    def save_tensor_to_wav(self, tensor, sample_rate, output_file):
        try:
            if tensor.numel() == 0:
                logger.warning(f"Attempted to save empty tensor to {output_file}. Skipping.")
                return
            if tensor.ndim == 1:
                tensor = tensor.unsqueeze(0)
            tensor_for_save = tensor.cpu().float()
            torchaudio.save(output_file, tensor_for_save, sample_rate, format="wav", bits_per_sample=16)
        except Exception as e:
            logger.error(f"Error saving tensor to WAV {output_file}: {e}", exc_info=True)