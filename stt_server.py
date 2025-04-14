import stt_pb2_grpc as stt__pb2__grpc

from concurrent import futures

import argparse
import logging
import asyncio
import grpc
from transcribe_server import TranscriptionServer
from google.cloud import texttospeech

FORMAT = '%(levelname)s: %(asctime)s: %(message)s'
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('SttServer')

class Listener(stt__pb2__grpc.ListenerServicer):
    def __init__(self, project, location) -> None:
        super().__init__()
        self.project = project
        self.location = location

    async def DoSpeechToText(self, request_iterator, context: grpc.aio.ServicerContext):
        '''Main rpc function for converting speech to text'''
        logger.info("do speech to text begins.")
        transcription_server = TranscriptionServer(
            project_id=self.project, location=self.location, recognizer="-")
        async for requests in request_iterator:
            print("request_iterator begins:")
            chunk = requests.content.audio
            language_code = requests.config.streaming_config.config.language_codes[0]
            task = asyncio.ensure_future(
                transcription_server.recv_audio_bytes(chunk, language_code)) #创建task
            result = await task # 立即等待task完成
            #print(f"xxxxx={result}")
            if result is not None:
                yield result # 立即yield结果
                for res in result.results:
                    is_final_status = res.is_final
                    for alt in res.alternatives:
                        translation_text = alt.translation
                        transcript_text = alt.transcript
                if is_final_status:
                    speech = text2speech(translation_text)
                    print(type(speech))
                    print(len(speech.audio_content))

                
                # result = result.results
                # print(result)
                # print( type(result))
                # if result['is_final'] :
                #     speech = text2speech(result[alternatives[translation]])
                #     print(type(speech))
                #     print(len(speech))
                
                #if it's final, do TTS


async def serve(port, project, location):
    with futures.ThreadPoolExecutor(max_workers=20) as executor:
        server = grpc.aio.server(executor)
        stt__pb2__grpc.add_ListenerServicer_to_server(Listener(project, location), server)
        server.add_insecure_port('[::]:%d' % port)
        await server.start()
        await server.wait_for_termination()


def text2speech(input_text):
    client = texttospeech.TextToSpeechClient()
    synthesis_input = texttospeech.SynthesisInput(text=input_text)
    voice = texttospeech.VoiceSelectionParams(
        language_code="en-US", ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
    )
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3
    )
    response = client.synthesize_speech(
      input=synthesis_input, voice=voice, audio_config=audio_config
    )
    
    return response
    
    

    



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SpeechToText service')
    parser.add_argument('-p', action='store', dest='port', type=int, default=9080,
                     help='port')
    parser.add_argument('-project', action='store', dest='project', type=str, default='',
                     help='project')
    parser.add_argument('-location', action='store', dest='location', type=str, default='us-central1',
                     help='location')
    args = parser.parse_args()
    print(f"start stt server with port = {args.port} project={args.project} location={args.location}")
    asyncio.run(serve(args.port, args.project, args.location))
