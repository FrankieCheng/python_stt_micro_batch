from google.cloud import translate as translate
from google.api_core.client_options import ClientOptions
from google.cloud.speech_v2 import SpeechClient
from google.cloud.speech_v2.types import cloud_speech, ExplicitDecodingConfig, StreamingRecognizeResponse
import concurrent.futures
from concurrent.futures import as_completed
import vertexai
from vertexai.generative_models import GenerativeModel, Part
import vertexai.preview.generative_models as generative_models

import stt_pb2 as stt__pb2
import logging
import torchaudio
import numpy as np
import base64
from datetime import datetime
import io
import torch
from vad import (VADIterator,read_audio)

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

    def __init__(self, project_id, location, recognizer):
        torch.set_num_threads(1)
        self.PROJECT_ID = project_id  
        self.LOCATION = location
        self.model = torch.jit.load('silero_vad/silero_vad.jit')
        # we will use model temp to do some temp job, for example speech detect/segment the audios and so on.
        self.model_temp = torch.jit.load('silero_vad/silero_vad.jit')
        self.all_chunks = torch.tensor([])
        self.vad_iterator = VADIterator(self.model, threshold=0.5)
        self.recognizer = recognizer
        self.last_end = 0
        self.last_start = 0
        self.all_transcriptions = []
        self.speech_threshold = 0.5
        self.transcript_stream_results = []

    # used for .wav files input
    def recv_audio(self,
                   new_chunk, language_code):
        # read the chunks, and convert the chunks to SAMPLING_RATE(as 16000 default.)
        if new_chunk == None:
            return ""
        current_chunks = read_audio(new_chunk, sampling_rate=self.SAMPLING_RATE)
        return self.process_new_chunks(current_chunks, language_code)

    # used for bytes input, and only float32 is supported.
    def recv_audio_bytes(self,
                   new_chunk, language_code):
        #try:
            logger.info(f"{type(new_chunk)} len={len(new_chunk)}")
            #read the chunks, and convert the chunks to SAMPLING_RATE(as 16000 default.)
            audio_array = np.frombuffer(new_chunk, dtype=np.float32)
            #logger.info(audio_array)
            current_chunks = torch.Tensor(audio_array)
            return self.process_new_chunks(current_chunks, language_code)
        # except Exception as e:
        #     print(e)
        #     return None

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
            return stt__pb2.TranscriptStreamResponse(
                speech_event_offset = current_transcript_segments[0]['start'],
                results = transcript_stream_results)
        else:
            return None

    # process new chunks
    def process_new_chunks(self,
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
                current_all_segments.append({'start':current_last_start, 'immediate':i + len(chunk)})
                break
            # vad invoke, find the end of speech/segment.
            speech_dict = self.vad_iterator(chunk, return_seconds=False)
            # if end of segment founds, and split audio into big segments, and then find the next one.
            if speech_dict and 'end' in speech_dict:
                logging.info(f"--------1 end dict founded last_start={current_last_start} last_end={speech_dict['end']} index={i}")
                current_all_segments.append({'start':current_last_start, 'end':speech_dict['end']})
                current_last_start = speech_dict['end']
                self.last_start = speech_dict['end']

        logger.info(f"--------before transcript begins. is_speech={has_new_speech}")
        #if no new speech detected, just return the transcription, otherwise, we should process those.

        #used for debug only to check the wav files.
        #self.save_tensor_to_wav(self.all_chunks, 16000, f"checkpoint_to_wav_{len(self.all_chunks)}.wav")
        #TODO: no speech in current chunk has some issues.
        if not has_new_speech:
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
        
            with concurrent.futures.ThreadPoolExecutor() as executor:
                # change the method transcribe/transcribe_by_gemini, and temp used transcribe_by_gemini instead of chirp_2.
                #transcribe_thread_submit = executor.submit(self.transcribe, transcripted_base64_content, language_code)
                transcribe_thread_submit = executor.submit(self.transcribe_by_gemini, transcripted_base64_content, LANGUAGE_CODE_DIC[language_code])
                for future in as_completed([transcribe_thread_submit]):
                    transcript_json_result = future.result()
                    
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
    def transcribe (
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
        response = client.recognize(request=audio_request)

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
    def transcribe_by_gemini(self, audio_base64,
        language):
        prompt_template = """<Transcription Instructions>
1.Faithful to Original: Accurately transcribe the audio content, including all spoken words.
2.Formal Language: Use standard {language}.
3.Remove all Onomatopoeia and Interjections: Such as "um," "ah," "oh," "啊", "삐", '哔', unless crucial for understanding the meaning.
4.Eliminate Mimetic Words: Such as "huālā lā" (sound of water), "dī dā dī dā" (ticking clock), remove them directly.
5.Attention to Proper Nouns and Terminology: Double-check for accuracy.
6.No Speech Detected: If no speech is detected in a segment, leave the transcription output blank for that segment.
7.If the audio file or a link to the audio file is not provided, output NULL instead of asking input.
8.Incomplete Sentences: If the audio contains incomplete sentences or phrases, transcribe what is heard without adding words or trying to complete the sentence.
</Transcription Instructions>
<Example>
Original Audio: "Uh... well, I just heard a 'bang,' it seems like something fell down, oh, I hope it's not the vase?" (followed by 10 seconds of silence)
Transcribed Text: "I just heard a sound, it seems like something fell down, I hope it's not the vase?" (blank space for the silent segment)
</Example>
<Additional Notes>
Punctuation: Use punctuation correctly based on pauses and intonation.
Special Cases: If encountering difficult content or no audio/content/input, please output blank or NULL.
</Additional Notes>
"""
        generation_config = {
            "max_output_tokens": 8192,
            "temperature": 0.1,
            "top_p": 0.95,
        }

        safety_settings = {
            generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
            generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
            generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
            generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_ONLY_HIGH,
        }

        start_time = (int)(datetime.now().timestamp() * 1000)
        vertexai.init(project=self.PROJECT_ID, location=self.LOCATION)
        model = GenerativeModel(
            "gemini-1.5-pro-001",
        )
        prompt_contents = [Part.from_data(mime_type="audio/wav",data=base64.b64decode(audio_base64)), prompt_template.format(language=language)]
        response = model.generate_content(
            prompt_contents,
            generation_config=generation_config,
            safety_settings=safety_settings,
            stream=False,
        )

        end_time = (int)(datetime.now().timestamp() * 1000)
        gemini_used_time = end_time - start_time
        logger.info(f"transcript by gemini start_time={start_time} end_time={end_time} gemini_used_time={gemini_used_time}, transcript={response.text}")
        return self.process_ununsed(response.text)

    #process the unused, only processed chinese/english now.
    def process_ununsed(self, txt):
        txt = txt.replace("\n", "").replace("`", "").replace("삐", "")
        txt = txt.replace("<spacing>","").replace("<noise>","").replace("<spoken_noise>", "")
        return txt.lower().replace("null", "").strip()
    
    # save all chunks into wav.
    def save_all(self):
        self.save_tensor_to_wav(self.all_chunks, self.SAMPLING_RATE, f"output_audio_all_1_{len(self.all_chunks)}.wav")