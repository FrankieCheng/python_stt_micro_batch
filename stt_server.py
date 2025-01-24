import stt_pb2
import stt_pb2_grpc as stt__pb2__grpc

from concurrent import futures

import argparse
import logging
import asyncio
import grpc

from TxtSpeechBridge import TextToSpeechBridge
from transcribe_server import TranscriptionServer

FORMAT = '%(levelname)s: %(asctime)s: %(message)s'
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('SttServer')

class Listener(stt__pb2__grpc.ListenerServicer):
    def __init__(self, project, location) -> None:
        super().__init__()
        self.project = project
        self.location = location

    async def DoSpeechToText(self, request_iterator, context: grpc.aio.ServicerContext):
        '''Main rpc function for converting speech to text
            Takes in a stream of stt_pb2 SpeechChunk messages
            and returns a stream of stt_pb2 Transcript messages
        '''
        logger.info("do speech to text begins.")
        transcription_server = TranscriptionServer(
            project_id=self.project, location=self.location, recognizer="-")
        # keep sending transcript to client until *all* ASRs are DONE
        async for requests in request_iterator:
            print("request_iterator begins:")
            chunk = requests.content.audio
            # temp use the first language code as the target language.
            language_code = requests.config.streaming_config.config.language_codes[0]

            transcript_result_task = asyncio.create_task(
                transcription_server.recv_audio_bytes(chunk, language_code))
            transcript_result = await transcript_result_task
            if None != transcript_result:
                yield transcript_result

    def txt_to_speech(self,request_iterator,context):
        bridge = TextToSpeechBridge(None)
        for request in request_iterator:
            text = request.text  # 从请求中获取文本内容
            audio_content = bridge.start(text)
            yield stt_pb2.AudioResponse(audio=audio_content)

async def serve(port, project, location):
    with futures.ThreadPoolExecutor(max_workers=20) as executor:
        server = grpc.aio.server(executor)
        stt__pb2__grpc.add_ListenerServicer_to_server(Listener(project, location), server)
        server.add_insecure_port('[::]:%d' % port)
        await server.start()
        await server.wait_for_termination()

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
