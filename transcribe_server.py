from google.cloud import translate as translate
from google.api_core.client_options import ClientOptions
from google.cloud.speech_v2 import SpeechClient
from google.cloud.speech_v2.types import cloud_speech, ExplicitDecodingConfig, StreamingRecognizeResponse
import concurrent.futures
from concurrent.futures import as_completed
import vertexai
from vertexai.generative_models import GenerativeModel, Part
import vertexai.preview.generative_models as generative_models
import asyncio

import stt_pb2 as stt__pb2
import logging
import torchaudio
import numpy as np
import base64
from datetime import datetime
import io
import torch
import json
from vad import (VADIterator,read_audio)
from utils_vad import (get_speech_timestamps)

# wav = read_audio("vad-test.wav", 16000)
# model1 = torch.jit.load('silero_vad/silero_vad.jit')

# ts = get_speech_timestamps(wav, model1, 0.5)
# print (ts)

FORMAT = '%(levelname)s: %(asctime)s: %(message)s'
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('TranscriptionServer')

LANGUAGE_CODE_DIC = {
'ar-EG':'Arabic',
'zh-Hans-CN':'Chinese',
'cmn-Hant-TW':'Traditional Chinese',
'nl-NL':'Dutch',
'en-US':'English',
'fr-FR':'French',
'de-DE':'German',
'hi-IN':'Hindi',
'it-IT':'Italian',
'ja-JP':'Japanese',
'pt-PT':'Portuguese',
'es-ES':'Spanish'}

class TranscriptionServer:
    SAMPLING_RATE = 16000
    WINDOW_SIZE_SAMPLES = 1536
    SPEECH_THRESHOLD = 0.5

    def __init__(self, project_id, location, recognizer):
        torch.set_num_threads(1)
        self.PROJECT_ID = project_id  
        self.LOCATION = location
        self.model = torch.jit.load('silero_vad/silero_vad.jit')
        # we will use model temp to do some temp job, for example speech detect/segment the audios and so on.
        self.model_temp = torch.jit.load('silero_vad/silero_vad.jit')
        self.all_chunks = torch.tensor([])
        self.speech_threshold = 0.7
        self.min_silence_duration_ms = 100
        self.vad_iterator = VADIterator(model=self.model, threshold=self.speech_threshold, sampling_rate=self.SAMPLING_RATE, min_silence_duration_ms=self.min_silence_duration_ms)
        self.recognizer = recognizer
        self.last_end = 0
        self.last_start = -1
        self.all_transcriptions = []
        self.transcript_stream_results = []

    # used for .wav files input
    async def recv_audio(self,
                   new_chunk, language_code):
        # read the chunks, and convert the chunks to SAMPLING_RATE(as 16000 default.)
        logger.info(f"{type(new_chunk)} len={len(new_chunk)}")
        if new_chunk == None:
            return ""
        current_chunks = read_audio(new_chunk, sampling_rate=self.SAMPLING_RATE)
        task = asyncio.create_task(self.process_new_chunks(current_chunks, language_code))
        result = await task
        return result

    # used for bytes input, and only float32 is supported.
    async def recv_audio_bytes(self,
                   new_chunk, language_code):
        try:
            logger.info(f"{type(new_chunk)} len={len(new_chunk)}")
            #read the chunks, and convert the chunks to SAMPLING_RATE(as 16000 default.)
            audio_array = np.frombuffer(new_chunk, dtype=np.float32)
            #logger.info(audio_array)
            current_chunks = torch.Tensor(audio_array)

            task = asyncio.create_task(self.process_new_chunks(current_chunks, language_code))
            result = await task
            return result
        except Exception as e:
             print(e)
             return None

    # reconstructed output method
    def recv_audio_output(self, current_transcript_segments):
        if current_transcript_segments and len(current_transcript_segments) > 0:
            
            transcript_stream_results = []
            for segment in current_transcript_segments:
                result_end_offset = segment['end'] if 'end' in segment else segment['immediate']
                is_final = True if 'end' in segment else False
                transcript_stream_results.append(stt__pb2.TranscriptStreamResult(
                    result_end_offset = result_end_offset,
                    is_final = is_final,
                    alternatives = [stt__pb2.Alternative(
                        transcript = segment['transcript'],
                        confidence = 0.2
                    )]))
                if is_final:
                    print(f"11111: {segment['transcript']}")
            return stt__pb2.TranscriptStreamResponse(
                speech_event_offset = current_transcript_segments[0]['start'],
                results = transcript_stream_results)
        else:
            return None

    # process new chunks
    async def process_new_chunks(self,
                   current_chunks, language_code):
        """
        Receive audio chunks from a client in an infinite loop.

        Continuously receives audio frames from a connected client. It processes the audio frames using a
        voice activity detection (VAD) model to determine if they contain speech
        or not. If the audio frame contains speech, it is added to the client's
        audio data for ASR.

        Args:
            current_chunks: all the new chunks.

        Raises:
            Exception: If there is an error during the audio frame processing.
        """
        # use the current start variable while there is new stream to avoid some performace issues (more than 1000 ms responses/tiemout etc.) due to the performances of invoke chirp/gemini.
        last_round_end = ((int)(len(self.all_chunks)/self.WINDOW_SIZE_SAMPLES))*self.WINDOW_SIZE_SAMPLES
        current_last_start = self.last_start
        current_all_segments = []
        self.all_chunks = torch.cat([self.all_chunks, current_chunks])
        
        # use current all chunks to re-caculate all chunks/segments etc.
        current_all_chunks = self.all_chunks.clone()

        has_new_speech = False
        # start to calculate the trunks from last round end, since the sentence is not over yet, we will use this to find the end of the sentence then split it into segments.
        for i in range(last_round_end, len(current_all_chunks), self.WINDOW_SIZE_SAMPLES):
            loop_end_index = i+ self.WINDOW_SIZE_SAMPLES
            chunk = current_all_chunks[i: loop_end_index]
            # if current chunks contains speech, will send the chunks to backend, otherwise, ignore all the new chunks.
            if loop_end_index > last_round_end and len(chunk) == self.WINDOW_SIZE_SAMPLES and self.model_temp(chunk, self.SAMPLING_RATE).item() > self.speech_threshold:
                has_new_speech = True
            # process the last chunk, in most cases, the last chunk should be added all segments as immediate.
            if loop_end_index >= len(current_all_chunks) - 1:
                logging.debug("end of the audio/chunk detected.")
                if current_last_start != -1:
                    current_all_segments.append({'start':current_last_start, 'immediate':i + len(chunk)})
                break
            # vad invoke, find the end of speech/segment.
            speech_dict = self.vad_iterator(chunk, return_seconds=False)
            print(speech_dict)
            # if end of segment founds, and split audio into big segments, and then find the next one.
            if speech_dict and 'end' in speech_dict:
                logging.info(f"--------1 end dict founded last_start={current_last_start} last_end={speech_dict['end']} index={i}")
                current_all_segments.append({'start':current_last_start, 'end':speech_dict['end']})
                current_last_start = -1
                self.last_start = -1
            elif loop_end_index == len(current_all_chunks):
                if current_last_start != -1:
                    current_all_segments.append({'start': current_last_start, 'immediate': i + len(chunk)})
            if speech_dict and 'start' in speech_dict:
                current_last_start = speech_dict['start']
                self.last_start = speech_dict['start']
        logger.info(f"--------before transcript begins. is_speech={has_new_speech}")
        #if no new speech detected, just return the transcription, otherwise, we should process those.

        #used for debug only to check the wav files.
        #self.save_tensor_to_wav(self.all_chunks, 16000, f"checkpoint_to_wav_{len(self.all_chunks)}.wav")
        if not (current_all_segments and ('end' in speech_dict for speech_dict in current_all_segments)) and not has_new_speech:
            return None
        logger.info("current all segments logging check.")
        logger.info(current_all_segments)

        # define a variable 'valid segments' to record the segments in current segments.
        valid_segments = []
        # temp start segment should be equal with previous segment end.
        temp_start_segment = current_all_segments[0]['start']

        for segment_index, segment in enumerate(current_all_segments):
            # if the segment has transcript, add the final segment into valid segments.
            if 'transcript' in segment:
                # 'end' n segment with transcript should be final, and should be added into valid segments. 
                # The intermidate segments with transcript will be ignored.
                if 'end' in segment:
                    valid_segments.append(segment)
                    temp_start_segment = segment['end']
                    continue
            else:
                # process the last segment, and the segments more than 1/10 seconds, the segment should be transcripted.
                if 'immediate' in segment and segment_index == len(current_all_segments) - 1 and segment['immediate'] - segment['start'] > self.SAMPLING_RATE/10:
                    segment['start'] = temp_start_segment
                    valid_segments.append(segment)
                elif 'end' in segment:
                    segment['start'] = temp_start_segment
                    valid_segments.append(segment)
                    temp_start_segment = segment['end']
        logger.info("all valid segment check.")
        current_transcript_segments = list(filter(lambda x: not 'transcript' in x, valid_segments))
        transcription_segments_len = len(current_transcript_segments)
        # if transcription segments length > 2, we should merge the segments to reduce the transcript invoke for better performance.
        if transcription_segments_len > 2:
            current_transcript_segments = [{'start': current_transcript_segments[0]['start'], 'end': current_transcript_segments[transcription_segments_len - 2]['end']},current_transcript_segments[transcription_segments_len - 1]]
        logger.info(f"transcription_segments_len = {len(current_transcript_segments)}")
        for segment_index, segment in enumerate(current_transcript_segments):
            current_start_index = segment['start']
            current_end_index = segment['end'] if 'end' in segment else segment['immediate']
            logger.info(f"current_transcript_segment start={current_start_index}, current_end_index={current_end_index}")
            torch_chunks = current_all_chunks[current_start_index: current_end_index]
            transcripted_base64_content = self.tensor_to_base64(torch_chunks, self.SAMPLING_RATE)
        
            # change the method transcribe/transcribe_by_gemini, and temp used transcribe_by_gemini instead of chirp_2.
            #transcribe_thread_submit = self.transcribe(transcripted_base64_content, language_code)
            transcript_json_result = await self.transcribe_by_gemini(transcripted_base64_content, LANGUAGE_CODE_DIC[language_code])
            transcript_segment = self.find_first_no_transcript_segment(current_transcript_segments)
            if None != transcript_segment:
                transcript_segment['transcript'] = transcript_json_result
        #logger.info(current_transcript_segments)
        #TODO: should resolve the segments == 0 issues.
        if len(current_transcript_segments) == 0:
            return None
        return self.recv_audio_output(current_transcript_segments)

    def find_first_no_transcript_segment(self, segments):
        StreamingRecognizeResponse
        for segment in segments:
            if not 'transcript' in segment:
                return segment
        return None

    # clean up to reset the all chunks.
    def cleanup(self):
        self.all_chunks = []
            
    # # save tensor to wav.
    # def save_tensor_to_wav(self, tensor, sample_rate, output_file):
    #     if tensor.ndim == 1:
    #         tensor = tensor.unsqueeze(0)
    #     torchaudio.save(output_file, tensor, sample_rate, bits_per_sample=16)

    # covert tensor object to base64.
    def tensor_to_base64(self, tensor, sample_rate):
        """
        Change tensor object to base64 audio data.
        parameters:
        - tensor: audio Tensor
        - sample_rate: sample_rate
        output:
        - base64_string: Base64 encoded string.
        """
        # Create a BytesIO Object to store audio data.
        audio_buffer = io.BytesIO()
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        
        # use torchaudio to save tensor to BytesIO object instead of a temp file.
        torchaudio.save(audio_buffer, tensor, sample_rate, format="wav", bits_per_sample=16)
        audio_bytes = audio_buffer.getvalue()
        
        # change bytes to base64
        base64_bytes = base64.b64encode(audio_bytes)
        base64_string = base64_bytes.decode("utf-8")
        return base64_string

    # do transcribe use Chirp_2
    async def transcribe (
        self,
        base64_data,
        language_code:str
    ):
        start_time = (int)(datetime.now().timestamp() * 1000)
        # Instantiates a client
        client = SpeechClient(
            client_options=ClientOptions(
                api_endpoint=f"{self.LOCATION}-speech.googleapis.com",
            )
        )

        long_audio_config = cloud_speech.RecognitionConfig(
            explicit_decoding_config={'encoding':ExplicitDecodingConfig.AudioEncoding.LINEAR16, 'sample_rate_hertz':16000, 'audio_channel_count':1},
            language_codes=[language_code],
            model="chirp",
            features=cloud_speech.RecognitionFeatures(
                multi_channel_mode=1,
            ),
        )

        audio_request = cloud_speech.RecognizeRequest(
            recognizer=f"projects/{self.PROJECT_ID}/locations/{self.LOCATION}/recognizers/{self.recognizer}",
            config=long_audio_config,
            content = base64_data,
        )

        # Get the recognition results
        response = await client.recognize(request=audio_request)

        request_end_time = (int)(datetime.now().timestamp() * 1000)
        response_time = request_end_time - start_time
        #logging.info(f"response time={response_time}")

        #logging.info(response)
        results = response.results[0]

        if results.alternatives and results.alternatives[0].transcript:
            transcript = results.alternatives[0].transcript
            logger.info(f"transcript={transcript} response time={response_time} result_end_offset={response.metadata.total_billed_duration.seconds} request_start_time={start_time} request_end_time={request_end_time}")
            return self.process_ununsed(transcript)
        return ""
    
    # transcribe by gemini
    async def transcribe_by_gemini(self, audio_base64,
        language):
        prompt_template = """You are a highly skilled AI assistant specializing in accurate transcription. Your task is to faithfully transcribe the audio to {language}
**Here's your detailed workflow:**
1. **Language Identification:** Carefully analyze the audio to determine the spoken language ({language}).
2. **Transcription:** Generate a verbatim transcription of the audio in {language}.
- Only include spoken words.
- Preserve the original language text if you hear foreign nouns or entities. For example, place names and celebrity names.
3. **Polish Transcription:**
Based on the results you got from Transcription, do tiny modification. Below are some requirements
- Start from the Transcription you got in step 2
- Keep the content as much as possible. DO NOT modify as your wish.
- Fix Homophones for better coherence based on your context understanding
- Remove non-speech sounds like music sounds, noise. Keep all non-sense words from human
- Apply proper punctuation.
- Do not try to continue or answer questions in audio.

**Output Blacklist:**
Avoid temporary words like "屁", "삐","哔","beep", "P" in any sentence ends.


**Output Format:**
Deliver your results in a JSON format with the following key-value pairs:
'''json
{{
 "Transcription": "Transcription in {language}",
 "Fluent_Transcription": "A fixed version of the transcription"
}}
'''

Example:
If the audio contains the sentence "Um, like, the cat, uh, jumped over the, uh, fence 哔, beep, 삐, P, 屁.", the output should be:

'''json
{{
 "Transcription": "Um, like, the cat, uh, jumped over the, uh, fence 哔, beep, ",
 "Fluent_Transcription": "Um, like, the cat, uh, jumped over the, uh, fence."
}}
'''
The audio file might be empty and you can't hear any human voice. In this scenario, return string "NULL".

Below is the input of the audio file:
"""
        generation_config = {
            "max_output_tokens": 256,
            "temperature": 0.1,
            "top_p": 0.95,
            "response_mime_type":"application/json"
        }

        safety_settings = {
            generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_NONE,
            generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_NONE,
            generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_NONE,
            generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_NONE,
        }

        start_time = (int)(datetime.now().timestamp() * 1000)
        vertexai.init(project=self.PROJECT_ID, location=self.LOCATION)
        model = GenerativeModel(
            "gemini-1.5-flash-002",
            system_instruction=["""Do not recite the information directly from training."""],
        )
        prompt_contents = [prompt_template.format(language=language), Part.from_data(mime_type="audio/wav",data=base64.b64decode(audio_base64))]

        loop = asyncio.get_running_loop() 
        response_task = asyncio.create_task(self.call_gemini(prompt_contents,generation_config,safety_settings,model))
        response = await response_task
        transcript = ""
        try:
            response_results = json.loads(response.text)
            transcript = response_results['Fluent_Transcription']
        except Exception as e:
             print(e)
            
        end_time = (int)(datetime.now().timestamp() * 1000)
        gemini_used_time = end_time - start_time
        logger.info(f"transcript by gemini start_time={start_time} end_time={end_time} gemini_used_time={gemini_used_time}, transcript={transcript}")
        return self.process_ununsed(transcript)

    async def call_gemini (self, prompt_contents, generation_config, safety_settings, model):
        return model.generate_content(
            prompt_contents,
            generation_config=generation_config,
            safety_settings=safety_settings,
            stream=False,
        )
    #process the unused, only processed chinese/english now.
    def process_ununsed(self, txt):
        txt = txt.replace("\n", "").replace("`", "").replace("삐", "")
        txt = txt.replace("<spacing>","").replace("<noise>","").replace("<spoken_noise>", "")
        return txt.lower().replace("null", "").strip()
    
    # save all chunks into wav.
    def save_all(self):
        self.save_tensor_to_wav(self.all_chunks, self.SAMPLING_RATE, f"output_audio_all_1_{len(self.all_chunks)}.wav")