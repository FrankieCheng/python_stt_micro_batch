from prompt import LANGUAGE_LIST,HOTWORDS,PROMPT_TRANSCRIPT,PROMPT_MUL_TRANSLATE,PROMPT_TRANSLATE,PROMPT_TRANSLATE_TO_TARGET
from google.api_core.client_options import ClientOptions
from google.cloud.speech_v2 import SpeechClient
from google.cloud.speech_v2.types import cloud_speech, ExplicitDecodingConfig, StreamingRecognizeResponse
import time
from pympler import muppy, summary
import vertexai
from vertexai.generative_models import GenerativeModel, Part
import vertexai.preview.generative_models as generative_models
import asyncio
from loguru import logger
import logging
import torchaudio
import numpy as np
import base64
from datetime import datetime
import io
import torch
torch.set_grad_enabled(False)
import json
from vad import (VADIterator, read_audio)

import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/home/liang_xudan/stt_micro_batch_0205/zte-gemini.json"
# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/home/ubuntu/python_stt_micro_batch-main/cloudplus1-test-new-f7f503ac2fd0.json"
# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./cloudplus1-test-new-f7f503ac2fd0.json"

LANGUAGE_CODE_DIC = {
    'ar-EG': 'Arabic',
    'zh-Hans-CN': 'Chinese',
    'cmn-Hant-TW': 'Traditional Chinese',
    'nl-NL': 'Dutch',
    'en-US': 'English',
    'fr-FR': 'French',
    'de-DE': 'German',
    'hi-IN': 'Hindi',
    'it-IT': 'Italian',
    'ja-JP': 'Japanese',
    'pt-PT': 'Portuguese',
    'es-ES': 'Spanish',
    'ko-KR': 'Korean'
}

import gc

PROCESS_MODE_TRANSLATE = 'translate'
PROCESS_MODE_TRANSCRIPT = 'transcript'
PROCESS_MODE_MUL_TRANSLATE = 'mul_translate'
PROCESS_MODE_TRANSLATE_TO_TARGET = 'translate_to_target'

PROMPT_DIC = {
    PROCESS_MODE_TRANSLATE: PROMPT_TRANSLATE,
    PROCESS_MODE_TRANSCRIPT: PROMPT_TRANSCRIPT,
    PROCESS_MODE_MUL_TRANSLATE: PROMPT_MUL_TRANSLATE,
    PROCESS_MODE_TRANSLATE_TO_TARGET: PROMPT_TRANSLATE_TO_TARGET
}

class TranscriptionServer:
    SAMPLING_RATE = 16000
    WINDOW_SIZE_SAMPLES = 1536
    # WINDOW_SIZE_SAMPLES = 512
    SPEECH_THRESHOLD = 0.5

    def __init__(self, project_id, location, recognizer, ws):
        torch.set_num_threads(1)
        torch._C._jit_set_profiling_executor(False)

        self.PROJECT_ID = project_id
        self.LOCATION = location

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.debug(self.device)
        self.model = torch.jit.load('silero_vad/silero_vad.jit', map_location=self.device).eval()
        self.model = self.model.to(self.device)
        self.model_temp = torch.jit.load('silero_vad/silero_vad.jit', map_location=self.device).eval()
        self.model_temp = self.model_temp.to(self.device)

        self.all_chunks = torch.tensor([])
        self.speech_threshold = 0.7
        self.vad_iterator = VADIterator(self.model, self.speech_threshold)
        self.recognizer = recognizer
        self.last_end = 0
        self.last_start = 0
        self.all_transcriptions = []
        self.transcript_stream_results = []
        self.transcript_history = []
        self._queue = asyncio.Queue()
        self.ws = ws



    async def __aenter__(self):
        # 在这里执行初始化操作
        logger.info(f"TranscriptionServer for {self.PROJECT_ID} initialized.")
        return self

    async def __aexit__(self,exc_type, exc_val, exc_tb):
        # 清理资源
        # self.all_chunks = torch.tensor([])
        self.all_transcriptions = []
        self.transcript_stream_results = []
        self.transcript_history = []
        self._queue = asyncio.Queue()

    async def clear(self):
        # 清理资源
        # self.all_chunks = torch.tensor([])
        self.save_all()
        self.all_transcriptions = []
        self.transcript_stream_results = []
        self.transcript_history = []
        del self.model_temp
        del self.model
        # torch.cuda.empty_cache()
        gc.collect()

        logger.debug("清理执行完成")

        await self.clear_queue()

    async def clear_queue(self):
        while not self._queue.empty():
            task = await self._queue.get()
            self._queue.task_done()

    async def get_item(self):
        item_list = []
        for i in range(5):
            item = await self._queue.get()  # 获取队列中的一个元素

            if item is None:  # 如果接收到 None，则终止
                continue
            item_list.append(item)
        return item_list


    async def start(self):
        """
        开始处理文本流并生成音频流。
        """
        try:
            merged_audio = bytearray()
            language_code = []
            total_duration = 0

            while True:  # 检查队列是否为空
                item = await self._queue.get()  # 获取队列中的一个元素

                if item is None:  # 如果接收到 None，则终止
                    continue

                # item_list = await self.get_item()
                #
                # for item in item_list:
                media = item["media"]
                payload = media["payload"]
                language_code = media["language_code"]
                chunk = base64.b64decode(payload)
                merged_audio.extend(chunk)

                if len(merged_audio) > 0:

                    chunk_duration = len(merged_audio) / 16000
                    total_duration += chunk_duration

                    # 输出当前音频流的总持续时间
                    # logger.info(f"当前音频流的总持续时间: {total_duration:.2f} 秒")
                    # logger.debug(f"queue_size:{self._queue.qsize()}")

                    audio_array = np.frombuffer(merged_audio, dtype=np.float32)
                    current_chunks = torch.Tensor(audio_array)
                    merged_audio = bytearray()
                    result = await self.process_new_chunks(current_chunks, language_code)
                    if result is not None:
                        await self.ws.send(json.dumps({"transcript": result}))

        except Exception as e:
            logger.error(f"Error in start: {e} {e.__traceback__.tb_lineno}")
        finally:
            merged_audio = bytearray()


    async def add_request(self, task):

        await self._queue.put(task)



    # used for .wav files input
    async def recv_audio(self,new_chunk, language_code):
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
            audio_array = np.frombuffer(new_chunk, dtype=np.float32)
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
                logger.debug(f"segment:{segment}")
                result_end_offset = segment['end'] if 'end' in segment else segment['immediate']
                is_final = True if 'end' in segment else False
                result_transcript = segment['transcript']
                result_translation = segment['translation']
                result_language = segment['language']

                last_result = self.transcript_history[-1] if self.transcript_history else {'is_final':False,'transcript':'','language':'','translation':''}
                last_transcript = last_result['transcript']

                if last_result['is_final'] == False and len(result_transcript) < len(last_transcript):
                    result_transcript = last_transcript

                result_data = {
                    'result_end_offset':result_end_offset,
                    'is_final':is_final,
                    'transcript':result_transcript,
                    'language':result_language,
                    'translate':result_translation
                }
                self.transcript_history.append(result_data)
                transcript_stream_results.append(result_data)
            return transcript_stream_results
        else:
            return None


    async def process_new_chunks(self,current_chunks, language_code):
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

        language_code_1 = ''
        language_code_2 = ''
        mode = ''
        if len(language_code) >= 3:
            language_code_1 = language_code[0]
            language_code_2 = language_code[2]
            mode = language_code[1]
        if len(language_code) == 2:
            language_code_1 = language_code[0]
            language_code_2 = LANGUAGE_CODE_DIC['en-US']
            mode = language_code[1]
        if len(language_code) == 1:
            language_code_1 = language_code[0]
            language_code_2 = LANGUAGE_CODE_DIC['en-US']
            mode = PROCESS_MODE_TRANSCRIPT
        if mode == '':
            mode = PROCESS_MODE_TRANSCRIPT
        if language_code_2 == '':
            language_code_2 = LANGUAGE_CODE_DIC['en-US']
        if language_code_1 == '':
            language_code_1 = LANGUAGE_CODE_DIC['zh-Hans-CN']

        logger.info("language_code_1: " + language_code_1 + "; language_code_2: " + language_code_2 + "; mode: " + mode)

        # use the current start variable while there is new stream to avoid some performace issues (more than 1000 ms responses/tiemout etc.) due to the performances of invoke chirp/gemini.
        last_round_end = ((int)(len(self.all_chunks) / self.WINDOW_SIZE_SAMPLES)) * self.WINDOW_SIZE_SAMPLES
        current_last_start = self.last_start
        current_all_segments = []
        self.all_chunks = torch.cat([self.all_chunks, current_chunks])

        # use current all chunks to re-caculate all chunks/segments etc.
        current_all_chunks = self.all_chunks.clone()

        has_new_speech = False
        # start to calculate the trunks from last round end, since the sentence is not over yet, we will use this to find the end of the sentence then split it into segments.

        for i in range(last_round_end, len(current_all_chunks), self.WINDOW_SIZE_SAMPLES):
            loop_end_index = i + self.WINDOW_SIZE_SAMPLES
            chunk = current_all_chunks[i: loop_end_index]
            if len(chunk) < self.WINDOW_SIZE_SAMPLES:
                padding_length = self.WINDOW_SIZE_SAMPLES - len(chunk)
                chunk = torch.cat([chunk, torch.zeros(padding_length)], dim=0)
            chunk = chunk.to(self.device)
            try:
                with torch.no_grad():
                    output = self.model_temp(chunk, self.SAMPLING_RATE).item()
                    gc.collect()
                    # torch.cuda.empty_cache()
                    # logger.debug(f"CPU memory usage: {psutil.virtual_memory().percent}%")  # 打印CPU内存信息
            except Exception as e:
                # logger.error(e)
                pass

            # if current chunks contains speech, will send the chunks to backend, otherwise, ignore all the new chunks.
            if loop_end_index > last_round_end and len(chunk) == self.WINDOW_SIZE_SAMPLES and output > self.speech_threshold:
                has_new_speech = True
            # process the last chunk, in most cases, the last chunk should be added all segments as immediate.
            if loop_end_index >= len(current_all_chunks) - 1:
                logging.debug("end of the audio/chunk detected.")
                if current_last_start != -1:
                    current_all_segments.append({'start': current_last_start, 'immediate': i + len(chunk)})
                break
            # vad invoke, find the end of speech/segment.


            try:
                with torch.no_grad():
                    speech_dict = self.vad_iterator(chunk, return_seconds=False)
                    # torch.cuda.empty_cache()
                    gc.collect()
            except:
                pass

            # speech_dict = self.vad_iterator(chunk, return_seconds=False)
            # logger.debug(f"speech_dict:{speech_dict}")

            # speech_dict = None
            # logger.debug(f"speech_dict:{speech_dict}")
            # if end of segment founds, and split audio into big segments, and then find the next one.
            if speech_dict and 'end' in speech_dict:
                logging.info(
                    f"--------1 end dict founded last_start={current_last_start} last_end={speech_dict['end']} index={i}")
                current_all_segments.append({'start': current_last_start, 'end': speech_dict['end']})
                current_last_start = -1
                self.last_start = -1
            elif loop_end_index == len(current_all_chunks):
                if current_last_start != -1:
                    current_all_segments.append({'start': current_last_start, 'immediate': i + len(chunk)})
            if speech_dict and 'start' in speech_dict:
                current_last_start = speech_dict['start']
                self.last_start = speech_dict['start']

            del speech_dict
            del chunk
            del output

        logger.info(f"--------before transcript begins. is_speech={has_new_speech}")
        # if no new speech detected, just return the transcription, otherwise, we should process those.

        # used for debug only to check the wav files.
        # self.save_tensor_to_wav(self.all_chunks, 16000, f"checkpoint_to_wav_{len(self.all_chunks)}.wav")
        if not (current_all_segments and ('end' in speech_dict for speech_dict in
                                          current_all_segments)) and not has_new_speech:
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
                if 'immediate' in segment and segment_index == len(current_all_segments) - 1 and segment['immediate'] - \
                        segment['start'] > self.SAMPLING_RATE * 2 / 10:
                    segment['start'] = temp_start_segment
                    valid_segments.append(segment)
                elif 'end' in segment:
                    # segment['start'] = temp_start_segment
                    valid_segments.append(segment)
                    temp_start_segment = segment['end']
        logger.info("all valid segment check.")
        current_transcript_segments = list(filter(lambda x: not 'transcript' in x, valid_segments))
        transcription_segments_len = len(current_transcript_segments)
        # if transcription segments length > 2, we should merge the segments to reduce the transcript invoke for better performance.
        if transcription_segments_len > 2:
            current_transcript_segments = [{'start': current_transcript_segments[0]['start'],
                                            'end': current_transcript_segments[transcription_segments_len - 2]['end']},
                                           current_transcript_segments[transcription_segments_len - 1]]
        logger.info(f"transcription_segments_len = {len(current_transcript_segments)}")


        async def _process_chunk(segment,language_code_1,language_code_2,mode):
            start_index = segment['start']
            end_index = segment['end'] if 'end' in segment else segment['immediate']
            logger.debug(f"start_index:{start_index} - end_index:{end_index}")

            torch_chunks = current_all_chunks[start_index:end_index]
            base64_content = self.tensor_to_base64(torch_chunks, self.SAMPLING_RATE)
            del torch_chunks

            # 调转录
            logger.debug("start transcribe_by_gemini")
            transcript_results = await self.transcribe_by_gemini(base64_content, language_code_1,language_code_2,mode)
            logger.debug("End transcribe_by_gemini")
            for item in transcript_results:
                logger.debug("transcribe: " + item + ": " + transcript_results[item])
            if 'transcript' in transcript_results:
                transcript_result = transcript_results['transcript']
            else:
                transcript_result = ""
            if 'translation' in transcript_results:
                translation_result = transcript_results['translation']
            else:
                translation_result = ""
            if 'language' in transcript_results:
                language_result = transcript_results['language']
            else:
                language_result = ""

            segment['transcript'] = transcript_result
            segment['translation'] = translation_result
            segment['language'] = language_result

            logger.debug(f"success transcript segment:{segment}")
            return segment

        # tasks = []
        tasks = [ asyncio.create_task(_process_chunk(segment, language_code_1,language_code_2,mode)) for segment in current_transcript_segments]
        # tasks.append(asyncio.create_task(_process_chunk(chunk, language_code)))
        logger.debug(f"{len(tasks)} 个task开始并发处理")
        # 等待所有的 VAD 和转录任务完成
        transcription_segments = await asyncio.gather(*tasks)
        all_chunks_size = torch.tensor([])
        return self.recv_audio_output(transcription_segments)


    def find_first_no_transcript_segment(self, segments):
        StreamingRecognizeResponse
        for segment in segments:
            if not 'transcript' in segment:
                return segment
        return None

    # clean up to reset the all chunks.
    def cleanup(self):
        # self.all_chunks = []
        pass


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
    async def transcribe(
            self,
            base64_data,
            language_code: str
    ):
        start_time = (int)(datetime.now().timestamp() * 1000)
        # Instantiates a client
        client = SpeechClient(
            client_options=ClientOptions(
                api_endpoint=f"{self.LOCATION}-speech.googleapis.com",
            )
        )

        long_audio_config = cloud_speech.RecognitionConfig(
            explicit_decoding_config={'encoding': ExplicitDecodingConfig.AudioEncoding.LINEAR16,
                                      'sample_rate_hertz': 16000, 'audio_channel_count': 1},
            language_codes=[language_code],
            model="chirp",
            features=cloud_speech.RecognitionFeatures(
                multi_channel_mode=1,
            ),
        )

        audio_request = cloud_speech.RecognizeRequest(
            recognizer=f"projects/{self.PROJECT_ID}/locations/{self.LOCATION}/recognizers/{self.recognizer}",
            config=long_audio_config,
            content=base64_data,
        )

        # Get the recognition results
        response = await client.recognize(request=audio_request)

        request_end_time = (int)(datetime.now().timestamp() * 1000)
        response_time = request_end_time - start_time
        # logging.info(f"response time={response_time}")

        # logging.info(response)
        results = response.results[0]

        if results.alternatives and results.alternatives[0].transcript:
            transcript = results.alternatives[0].transcript
            logger.info(
                f"transcript={transcript} response time={response_time} result_end_offset={response.metadata.total_billed_duration.seconds} request_start_time={start_time} request_end_time={request_end_time}")
            return self.process_ununsed(transcript)
        return ""

    # transcribe by gemini
    async def transcribe_by_gemini(self, audio_base64,
                                   language_code_1=LANGUAGE_CODE_DIC['zh-Hans-CN'], language_code_2=LANGUAGE_CODE_DIC['en-US'], mode=PROCESS_MODE_TRANSCRIPT):
        logger.debug(
            "language_code_1: " + language_code_1 + "; language_code_2: " + language_code_2 + "; mode: " + mode)


        generation_config = {
            "max_output_tokens": 2048,
            "temperature": 0.1,
            "top_p": 0.95,
            "response_mime_type": "application/json"
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

        if mode in PROMPT_DIC:
            prompt_template = PROMPT_DIC[mode]
            logger.debug("mode in PROMPT_DIC")
        else:
            prompt_template = PROMPT_DIC[PROCESS_MODE_TRANSCRIPT]
            logger.debug("mode not in PROMPT_DIC")
        if mode == PROCESS_MODE_TRANSLATE:
            logger.debug("mode is PROCESS_MODE_TRANSLATE")
            prompt_contents = [prompt_template.format(language_code_1=language_code_1, language_code_2=language_code_2,
                                                      hotwords=HOTWORDS),
                               Part.from_data(mime_type="audio/wav", data=base64.b64decode(audio_base64))]
        elif mode == PROCESS_MODE_MUL_TRANSLATE:
            logger.debug("mode is PROCESS_MODE_MUL_TRANSLATE")
            prompt_contents = [prompt_template.format(language_code_1=language_code_1, language_code_2=language_code_2,
                                                      hotwords=HOTWORDS),
                               Part.from_data(mime_type="audio/wav", data=base64.b64decode(audio_base64))]
        elif mode == PROCESS_MODE_TRANSLATE_TO_TARGET:
            logger.debug("mode is PROCESS_MODE_TRANSLATE_TO_TARGET")
            prompt_contents = [
                prompt_template.format(language_code_1=language_code_1, hotwords=HOTWORDS, language_list=LANGUAGE_LIST),
                Part.from_data(mime_type="audio/wav", data=base64.b64decode(audio_base64))]
        else:
            prompt_contents = [prompt_template.format(language_code_1=language_code_1, hotwords=HOTWORDS),
                               Part.from_data(mime_type="audio/wav", data=base64.b64decode(audio_base64))]

        time1 = (int)(datetime.now().timestamp() * 1000)
        response_task = asyncio.create_task(
            self.call_gemini(prompt_contents, generation_config, safety_settings, model))
        response = await response_task
        time2 = (int)(datetime.now().timestamp() * 1000)
        logger.debug(f"=========={time2-time1}")
        transcript = ""
        translation = ""
        language = ""
        try:
            response_results = json.loads(response.text)
            if 'language' in response_results:
                language = response_results['language']
            if 'Fluent_Transcription' in response_results:
                transcript = response_results['Fluent_Transcription']
            if 'Translation' in response_results:
                translation = response_results['Translation']
            logger.debug(f"transcription: {transcript}, translation: {translation}, language: {language}")
            logger.debug('end  print response results')
        except Exception as e:
            logger.error(str(e))

        end_time = (int)(datetime.now().timestamp() * 1000)
        gemini_used_time = end_time - start_time
        logger.info(
            f"transcript by gemini start_time={start_time} end_time={end_time} gemini_used_time={gemini_used_time}, transcript={transcript}")
        transcript = self.process_ununsed(transcript)
        translation = self.process_ununsed(translation)
        logger.debug("htb: after ununsed transcription: " + transcript + ";  translation: " + translation)
        transcript_results = {'transcript': transcript, 'translation': translation, 'language': language}
        return transcript_results

    async def call_gemini(self, prompt_contents, generation_config, safety_settings, model):
        # return json.dumps({"Transcription": "Um, like, the cat, uh, jumped over the, uh, fence 哔, beep, ",
 # "Fluent_Transcription": "Um, like, the cat, uh, jumped over the, uh, fence."})

        start_time = (int)(datetime.now().timestamp() * 1000)
        result = model.generate_content(
            prompt_contents,
            generation_config=generation_config,
            safety_settings=safety_settings,
            stream=False,
        )
        end_time = (int)(datetime.now().timestamp() * 1000)
        logger.debug(f"use_time:{end_time-start_time}")
        return result

    # process the unused, only processed chinese/english now.
    def process_ununsed(self, txt):
        txt = txt.replace("\n", "").replace("`", "").replace("삐", "")
        txt = txt.replace("<spacing>", "").replace("<noise>", "").replace("<spoken_noise>", "")
        return txt.lower().replace("null", "").strip()

    async def save_tensor_to_wav(self, tensor, sample_rate, output_file):
        if tensor.ndim == 1:
            tensor = tensor.unsqueeze(0)
        torchaudio.save(output_file, tensor, sample_rate, bits_per_sample=16)

    # save all chunks into wav.
    async def save_all(self):
        await self.save_tensor_to_wav(self.all_chunks, self.SAMPLING_RATE, f"/home/liang_xudan/python_stt_server_dev/wav/output_audio_all_1_{len(self.all_chunks)}.wav")
