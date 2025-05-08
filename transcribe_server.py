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

asr_model_name_gemini = "gemini-1.5-flash-001" # Changed back from 2.0-flash-lite-001 as per your current file top
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
    # ... (implementation as you provided - this is good) ...
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
    WINDOW_SIZE_SAMPLES = 1024 # Your current setting
    SPEECH_THRESHOLD = 0.33    # Your current setting

    def __init__(self, project_id, location, recognizer_id_str):
        # ... (implementation as you provided - client init, VAD init, etc.) ...
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
        self.min_silence_duration_ms = 400
        self.vad_iterator = VADIterator(model=self.vad_model, threshold=self.vad_speech_threshold_iterator,
                                        sampling_rate=self.SAMPLING_RATE,
                                        min_silence_duration_ms=self.min_silence_duration_ms)
        self.last_end = 0
        self.last_start = -1

    async def recv_audio_bytes(self, new_chunk, language_code):
        # ... (implementation as you provided, with normalization) ...
        try:
            logger.info(f"Type={type(new_chunk)}, len={len(new_chunk)}, Lang={language_code}") # Kept your original log format
            audio_array = np.frombuffer(new_chunk, dtype=np.float32)
            current_chunks_tensor = torch.from_numpy(audio_array.copy())
            if current_chunks_tensor.numel() > 0: 
                max_val = torch.abs(current_chunks_tensor).max()
                if max_val > 0: 
                    current_chunks_tensor = current_chunks_tensor / max_val
                    logger.debug(f"Normalized incoming audio chunk. Original max_val: {max_val:.4f}")
            result = await self.process_new_chunks(current_chunks_tensor, language_code)
            return result
        except Exception as e:
            logger.error(f"Error in recv_audio_bytes: {e}", exc_info=True)
            return None


    def recv_audio_output(self, current_transcript_segments):
        # ... (implementation as you provided - this populates protobuf correctly) ...
        if current_transcript_segments and len(current_transcript_segments) > 0:
            transcript_stream_results_list = []
            # Ensure 'start' exists, default to 0 if not (though VAD should always provide it)
            first_start_offset = current_transcript_segments[0].get('start', 0) 

            for segment in current_transcript_segments:
                result_end_offset_val = segment.get('end', segment.get('immediate', 0))
                is_final_val = 'end' in segment # Key 'end' marks a VAD-finalized segment
                transcript_val = segment.get('transcript', "")
                translation_val = segment.get('translation', "")
                confidence_val = segment.get('confidence', 0.9) # Default confidence if not provided
                stt_duration_val = segment.get('stt_duration', 0)
                translation_duration_val = segment.get('translation_duration', 0)

                alternatives_list = [stt__pb2.Alternative(
                    transcript=transcript_val,
                    translation=translation_val,
                    confidence=confidence_val,
                    stt_duration_ms=stt_duration_val,
                    translation_duration_ms=translation_duration_val
                    # tts_duration_ms is set by stt_server.py
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
            logger.debug("recv_audio_output called with no segments to output.")
            return None

    async def _process_single_segment_concurrently(self, segment_dict_copy, audio_b64_content, language_code, target_gemini_language_for_ast):
        current_start_index = segment_dict_copy['start']
        logger.info(f"Async task started for segment: start={current_start_index}, end={segment_dict_copy.get('end', segment_dict_copy.get('immediate'))}")

        # Initialize fields in the copy
        segment_dict_copy['stt_duration'] = 0
        segment_dict_copy['translation_duration'] = 0
        segment_dict_copy['transcript'] = ""
        segment_dict_copy['translation'] = ""

        try:
            # Perform transcription
            transcript_text, stt_duration = await self.transcribe_by_gemini(audio_b64_content, target_gemini_language_for_ast)
            segment_dict_copy['transcript'] = transcript_text
            segment_dict_copy['stt_duration'] = stt_duration
            logger.info(f"Segment transcript (async): '{transcript_text}' (STT: {stt_duration}ms) for start={current_start_index}")

            # Perform translation if the segment is final ('end' key exists) and transcription was successful
            if 'end' in segment_dict_copy and transcript_text:
                logger.info(f"Segment (async) start={current_start_index} is final, attempting translation. Source: {target_gemini_language_for_ast}, Target: {TARGET_LANGUAGE}")
                if target_gemini_language_for_ast != TARGET_LANGUAGE:
                    translation_text, translation_duration = await self.transcribe_and_translate_by_gemini(
                        audio_b64_content,
                        target_gemini_language_for_ast,
                        TARGET_LANGUAGE
                    )
                    segment_dict_copy['translation'] = translation_text
                    segment_dict_copy['translation_duration'] = translation_duration
                    logger.info(f"Segment translation (async): '{translation_text}' (Translate: {translation_duration}ms) for start={current_start_index}")
                else:
                    segment_dict_copy['translation'] = transcript_text
                    segment_dict_copy['translation_duration'] = 0
                    logger.info(f"Source and target language same for translation ('{TARGET_LANGUAGE}') for start={current_start_index}.")
            # elif 'end' in segment_dict_copy and not transcript_text: # This case already logged if transcript is empty
            #     logger.warning(f"Segment (async) start={current_start_index} is final but no transcript to translate.")
            
        except Exception as e:
            logger.error(f"Error processing segment (async) start={current_start_index}: {e}", exc_info=True)
            segment_dict_copy['transcript'] = segment_dict_copy.get('transcript', "[STT_ERROR]") # Keep partial if any
            segment_dict_copy['translation'] = segment_dict_copy.get('translation', "[TRANSLATE_ERROR]")
        
        return segment_dict_copy

    async def process_new_chunks(self, current_chunks, language_code):
        # --- VAD and Segment Identification Logic (as per your provided code) ---
        last_round_end = ((int)(len(self.all_chunks)/self.WINDOW_SIZE_SAMPLES))*self.WINDOW_SIZE_SAMPLES
        current_last_start = self.last_start
        current_all_segments = [] 
        self.all_chunks = torch.cat([self.all_chunks, current_chunks])
        current_all_chunks_for_vad = self.all_chunks.clone()
        has_new_speech = False

        for i in range(last_round_end, len(current_all_chunks_for_vad), self.WINDOW_SIZE_SAMPLES):
            loop_end_index = i + self.WINDOW_SIZE_SAMPLES
            chunk_for_vad_iter = current_all_chunks_for_vad[i: loop_end_index]
            
            if chunk_for_vad_iter.numel() < self.WINDOW_SIZE_SAMPLES and loop_end_index < len(current_all_chunks_for_vad):
                 # Not a full window, and not the very last part of all_chunks.
                 # VAD might behave unpredictably. Often VADs expect fixed size windows.
                 # Silero VAD can handle variable last chunk, but this direct check might be problematic.
                 logger.debug(f"Skipping direct VAD check for incomplete window of size {len(chunk_for_vad_iter)}")
            elif loop_end_index > last_round_end and len(chunk_for_vad_iter) == self.WINDOW_SIZE_SAMPLES : # Only process full windows for direct check
                if self.vad_model(chunk_for_vad_iter, self.SAMPLING_RATE).item() > self.SPEECH_THRESHOLD:
                    has_new_speech = True
            
            if loop_end_index >= len(current_all_chunks_for_vad) -1: 
                logging.debug("End of the audio buffer reached in VAD loop.")
                if current_last_start != -1: 
                    current_all_segments.append({'start': current_last_start, 'immediate': i + len(chunk_for_vad_iter)})
                break
            
            speech_dict = self.vad_iterator(chunk_for_vad_iter, return_seconds=False)
            if speech_dict: 
                if 'end' in speech_dict:
                    logging.info(f"VADIterator found 'end': current_last_start={current_last_start}, end_offset={speech_dict['end']}")
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
        
        logger.info(f"VAD processing complete. has_new_speech flag: {has_new_speech}, VAD-iterator detected segments: {len(current_all_segments)}")

        if not current_all_segments and not has_new_speech:
            logger.debug("No new speech segments detected by VAD this pass.")
            return None
        
        # +++ REINSTATE DEBUG SAVING for "to_website" (entire current buffer for VAD) +++
        # This saves the `current_all_chunks_for_vad` which is `self.all_chunks.clone()` at this point.
        # It represents the total audio buffer the VAD decisions below are based on for this pass.
        try:
            debug_audio_dir_vad_input = "debug_audio_vad_input" # New directory for clarity
            if not os.path.exists(debug_audio_dir_vad_input):
                os.makedirs(debug_audio_dir_vad_input)
                logger.info(f"Created directory for VAD input debug audio: {debug_audio_dir_vad_input}")
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            vad_input_filename = os.path.join(debug_audio_dir_vad_input, f"vad_input_buffer_{timestamp_str}_len{len(current_all_chunks_for_vad)}.wav")
            if current_all_chunks_for_vad.numel() > 0 : # Only save if not empty
                 self.save_tensor_to_wav(current_all_chunks_for_vad, self.SAMPLING_RATE, vad_input_filename)
                 logger.info(f"Saved current VAD input buffer to: {vad_input_filename}")
        except Exception as save_e:
            logger.error(f"Failed to save VAD input buffer debug audio: {save_e}", exc_info=True)
        # +++ END DEBUG SAVING +++

        # --- Your existing segment filtering logic ---
        segments_to_process_further = [] # Renamed for clarity
        temp_start_for_filtering = 0
        if current_all_segments:
             temp_start_for_filtering = current_all_segments[0].get('start',0)
        for idx, seg_from_vad in enumerate(current_all_segments):
            is_last_segment_from_vad = (idx == len(current_all_segments) - 1)
            if 'end' in seg_from_vad:
                seg_start = seg_from_vad.get('start', temp_start_for_filtering) 
                if seg_start < seg_from_vad['end']: 
                    segments_to_process_further.append({'start': seg_start, 'end': seg_from_vad['end']})
                    temp_start_for_filtering = seg_from_vad['end']
            elif 'immediate' in seg_from_vad and is_last_segment_from_vad: # Only process last 'immediate'
                seg_start = seg_from_vad.get('start', temp_start_for_filtering)
                # Use your defined min length for immediate results
                if seg_start < seg_from_vad['immediate'] and (seg_from_vad['immediate'] - seg_start > self.SAMPLING_RATE * 0.4): 
                    segments_to_process_further.append({'start': seg_start, 'immediate': seg_from_vad['immediate']})
        
        logger.info(f"Filtered segments for ASR/AST (pre-consolidation): {segments_to_process_further}")
        if not segments_to_process_further:
            logger.debug("No segments for ASR/AST after initial filtering.")
            return None
        
        # --- Your existing consolidation logic ---
        if len(segments_to_process_further) > 2: # Consolidate if more than 2 segments to reduce API calls
            logger.info(f"Consolidating {len(segments_to_process_further)} segments for transcription.")
            # Keep the very last segment (which might be an 'immediate' one) separate.
            # Consolidate all segments before the last one if they are 'end' segments.
            last_segment_to_keep = segments_to_process_further[-1]
            segments_to_consolidate = segments_to_process_further[:-1]
            
            if segments_to_consolidate: # Ensure there's something to consolidate
                consolidated_start = segments_to_consolidate[0]['start']
                # Find the 'end' of the last segment in the group to be consolidated
                consolidated_end = segments_to_consolidate[-1]['end'] if 'end' in segments_to_consolidate[-1] else segments_to_consolidate[-1]['immediate']

                final_segments_for_asr = [{'start': consolidated_start, 'end': consolidated_end}]
                final_segments_for_asr.append(last_segment_to_keep)
            else: # Only one segment was in segments_to_process_further (after being the last_segment_to_keep) or none
                 final_segments_for_asr = [last_segment_to_keep] if len(segments_to_process_further) == 1 else []


            logger.info(f"Consolidated into {len(final_segments_for_asr)} segments: {final_segments_for_asr}")
        else:
            final_segments_for_asr = segments_to_process_further # Use as is if 0, 1 or 2 segments

        if not final_segments_for_asr:
            logger.debug("No segments to submit for ASR after consolidation.")
            return None
        # --- End of VAD and segment identification/consolidation logic ---

        asr_processing_tasks = []
        for segment_info_for_task in final_segments_for_asr:
            current_start_index = segment_info_for_task['start']
            # Ensure 'immediate' is also considered if 'end' is not present for the last segment
            current_end_index = segment_info_for_task.get('end', segment_info_for_task.get('immediate')) 

            if current_end_index is None or current_end_index <= current_start_index:
                logger.warning(f"Skipping invalid segment for ASR task: start={current_start_index}, end={current_end_index}")
                continue

            logger.info(f"Preparing ASR task for segment: start={current_start_index}, end={current_end_index}")
            torch_segment_chunks = self.all_chunks[current_start_index:current_end_index]
            
            if torch_segment_chunks.numel() == 0:
                logger.warning(f"ASR Task: Segment resulted in empty torch_chunks: start={current_start_index}, end={current_end_index}. Creating dummy task.")
                async def empty_segment_task_handler(seg_dict_ref): # Modifies dict in place
                    seg_dict_ref['transcript'] = ""
                    seg_dict_ref['stt_duration'] = 0
                    seg_dict_ref['translation'] = ""
                    seg_dict_ref['translation_duration'] = 0
                    return seg_dict_ref
                task = asyncio.create_task(empty_segment_task_handler(segment_info_for_task.copy())) # Process a copy
                asr_processing_tasks.append(task)
                continue

            # Debug save for audio sent to Gemini
            try:
                debug_audio_dir_gemini = "debug_audio_clips_gemini" # Separate dir for clarity
                if not os.path.exists(debug_audio_dir_gemini): os.makedirs(debug_audio_dir_gemini)
                timestamp_str_gemini = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                filename_gemini = os.path.join(debug_audio_dir_gemini, f"to_gemini_{timestamp_str_gemini}_s{current_start_index}_e{current_end_index}.wav")
                self.save_tensor_to_wav(torch_segment_chunks, self.SAMPLING_RATE, filename_gemini)
                logger.info(f"Saved debug audio segment for Gemini to: {filename_gemini}")
            except Exception as save_e:
                logger.error(f"Failed to save debug audio segment for Gemini: {save_e}", exc_info=True)

            transcripted_base64_content = self.tensor_to_base64(torch_segment_chunks, self.SAMPLING_RATE)
            target_gemini_language_for_ast = LANGUAGE_CODE_DIC.get(language_code, "English")
            
            task = asyncio.create_task(
                self._process_single_segment_concurrently(
                    segment_info_for_task.copy(), # Pass a copy to the async task
                    transcripted_base64_content,
                    language_code, 
                    target_gemini_language_for_ast
                )
            )
            asr_processing_tasks.append(task)

        if not asr_processing_tasks:
            logger.debug("No ASR tasks were created for this pass.")
            return None
            
        logger.info(f"Awaiting completion of {len(asr_processing_tasks)} concurrent ASR/AST tasks...")
        output_segments_with_results = await asyncio.gather(*asr_processing_tasks)
        logger.info(f"All {len(asr_processing_tasks)} ASR/AST tasks completed.")
        output_segments_with_results = [res for res in output_segments_with_results if res is not None]

        if not output_segments_with_results:
            logger.debug("No segments produced results after concurrent ASR/AST processing.")
            return None
            
        return self.recv_audio_output(output_segments_with_results)

    def tensor_to_base64(self, tensor, sample_rate):
        audio_buffer = io.BytesIO()
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        # Ensure tensor is on CPU and float32 for torchaudio.save
        tensor_for_save = tensor.cpu().float()
        torchaudio.save(audio_buffer, tensor_for_save, sample_rate, format="wav", bits_per_sample=16)
        audio_bytes = audio_buffer.getvalue()
        base64_bytes = base64.b64encode(audio_bytes)
        base64_string = base64_bytes.decode("utf-8")
        return base64_string

    async def transcribe(self, base64_data_wav, language_code: str):
        if not self.speech_v2_client:
            logger.error("Chirp STT: SpeechClient (v2) not initialized. Cannot transcribe.")
            return "", 0 # Return text and duration
        
        start_time_ms = int(datetime.now().timestamp() * 1000)
        logger.info(f"Chirp STT: Transcribing for lang '{language_code}'. Data length (b64): {len(base64_data_wav)}")
        # ... (rest of your transcribe logic as in Response #27) ...
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
        duration_ms = 0 # Initialize duration
        try:
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(None, functools.partial(self.speech_v2_client.recognize, request=request))
            end_time_ms = int(datetime.now().timestamp() * 1000)
            duration_ms = end_time_ms - start_time_ms # Calculate duration
            logger.info(f"Chirp STT: Response received in {duration_ms} ms.")
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
        return transcript_text, duration_ms # Return tuple

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
        # ... (implementation as you provided, it already returns (text, duration_ms)) ...
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
        if response is None:
             logger.warning("Gemini ASR: call_gemini returned None.")
        elif hasattr(response, 'text'):
            # logger.info(f"Gemini Call transcribe: Response received: {response.text[:200]}...") # Log snippet
            try:
                response_results = json.loads(response.text)
                transcript = response_results.get('Fluent_Transcription', "")
            except json.JSONDecodeError:
                logger.error(f"Gemini ASR: Failed to parse JSON from response: {response.text}")
            except Exception as e:
                logger.error(f"Gemini ASR: Error processing Gemini response: {e}", exc_info=True)
        else:
            logger.warning(f"Gemini ASR: No valid 'text' attribute in response. Type: {type(response)}")
        end_time_ms = int(datetime.now().timestamp() * 1000)
        duration_ms = end_time_ms - start_time_ms
        logger.info(f"Gemini ASR: Transcript='{transcript}', Duration={duration_ms}ms")
        return self.process_ununsed(transcript), duration_ms

    async def transcribe_and_translate_by_gemini(self, audio_base64_wav, source_language_name, target_language_name):
        # ... (implementation as you provided, it already returns (text, duration_ms)) ...
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
            logger.warning("Gemini AST: call_gemini returned None.")
        elif hasattr(response, 'text'):
            # logger.info(f"Gemini Call translate: Response received: {response.text[:200]}...") # Log snippet
            try:
                response_results = json.loads(response.text)
                translation = response_results.get('Translation', "")
            except json.JSONDecodeError:
                logger.error(f"Gemini AST: Failed to parse JSON from response: {response.text}")
            except Exception as e:
                logger.error(f"Gemini AST: Error processing Gemini response: {e}", exc_info=True)
        else:
            logger.warning(f"Gemini AST: No valid 'text' attribute in response. Type: {type(response)}")
        end_time_ms = int(datetime.now().timestamp() * 1000)
        duration_ms = end_time_ms - start_time_ms
        logger.info(f"Gemini AST: Translation='{translation}', Duration={duration_ms}ms")
        return self.process_ununsed(translation), duration_ms

    def process_ununsed(self, txt):
        # ... (implementation as you provided) ...
        if not isinstance(txt, str):
            logger.warning(f"process_ununsed expected string, got {type(txt)}. Returning empty string.")
            return ""
        txt = txt.replace("\n", "").replace("`", "").replace("삐", "")
        txt = txt.replace("<spacing>","").replace("<noise>","").replace("<spoken_noise>", "")
        return txt.lower().replace("null", "").strip()
    
    def cleanup(self):
        # ... (implementation as you provided) ...
        logger.info("Cleaning up TranscriptionServer state for a new stream.")
        self.all_chunks = torch.tensor([])
        self.last_end = 0
        self.last_start = -1
        if hasattr(self.vad_iterator, 'reset_states') and callable(self.vad_iterator.reset_states):
            self.vad_iterator.reset_states()

    def save_tensor_to_wav(self, tensor, sample_rate, output_file):
        # ... (implementation as you provided, with empty tensor check) ...
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

    def find_first_no_transcript_segment(self, segments): # This seems unused now.
        # ... (implementation as you provided, removed stray comment) ...
        for segment in segments:
            if not 'transcript' in segment:
                return segment
        return None