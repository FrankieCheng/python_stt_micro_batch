import asyncio
import base64
import wave
import numpy as np
import websockets
import json
from loguru import logger
import pyaudio
import os

# 音频参数
SAMPLING_RATE = int(os.getenv('SAMPLING_RATE', 16000))
CHUNK = int(os.getenv('CHUNK', SAMPLING_RATE / 2))
# CHUNK = 4000
TIMEOUT = 1000
# URI = os.getenv('SERVER_URI', "ws://ec2-122-248-254-86.ap-southeast-1.compute.amazonaws.com:9080/chat")
URI = os.getenv('SERVER_URI', "ws://35.198.221.38:9081/chat") # 新加坡
# URI = os.getenv('SERVER_URI', "ws://34.123.219.129:9081/chat") # 美国
# URI = os.getenv('SERVER_URI', "ws://0.0.0.0:9081/chat")
MAX_RETRIES = 3


class MicrophoneStream:
    """异步音频流处理，使用asyncio.Queue避免阻塞事件循环"""

    def __init__(self, rate=SAMPLING_RATE, chunk=CHUNK):
        self._rate = rate
        self._chunk = chunk
        self._loop = asyncio.get_event_loop()
        self._queue = asyncio.Queue()
        self._audio_interface = None
        self._audio_stream = None
        self.closed = True

    def __enter__(self):
        self._audio_interface = pyaudio.PyAudio()
        self._audio_stream = self._audio_interface.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=self._rate,
            input=True,
            frames_per_buffer=self._chunk,
            stream_callback=self._audio_callback,
        )
        self.closed = False
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.closed = True
        self._audio_stream.stop_stream()
        self._audio_stream.close()
        self._audio_interface.terminate()
        self._loop.call_soon_threadsafe(self._queue.put_nowait, None)

    def _audio_callback(self, in_data, frame_count, time_info, status_flags):
        """音频采集回调（在音频线程执行）"""
        self._loop.call_soon_threadsafe(self._queue.put_nowait, in_data)
        return None, pyaudio.paContinue

    async def async_generator(self):
        """异步生成器"""
        while not self.closed:
            data = await self._queue.get()
            if data is None:
                return
            yield data


async def receive_responses(websocket):
    """实时处理服务端响应"""
    try:
        async for message in websocket:
            try:
                data = json.loads(message)
                if "transcript" in data and data["transcript"] is not None:
                    logger.info(f"实时转录: {data['transcript']}")
                elif "error" in data:
                    logger.error(f"服务端错误: {data['error']}")
                elif 'event' in data:
                    logger.debug(f"接收 ping-pong 响应")
            except json.JSONDecodeError:
                logger.error("无效的JSON响应")
    except websockets.exceptions.ConnectionClosed:
        logger.warning("服务端连接已关闭")


async def send_audio_chunks(websocket):
    """异步发送音频数据"""
    # with MicrophoneStream() as stream:
    #     await websocket.send(json.dumps({"event": "start"}))
    #     logger.info("录音启动...")
    #
    #     async for chunk in stream.async_generator():
    #         encoded = base64.b64encode(chunk).decode()
    #         await websocket.send(json.dumps({
    #             "event": "media",
    #             "media": {
    #                 "payload": encoded,
    #                 "language_code": "zh-Hans-CN"
    #             }
    #         }))
    #
    #     await websocket.send(json.dumps({"event": "stop"}))
    #     logger.info("录音结束...")

    def save_audio_to_wav(audio_data, sample_rate, filename):
        """
        将音频流的byte数据写入WAV文件。

        :param audio_data: byte数据，应该是原始的PCM音频数据
        :param sample_rate: 采样率，例如16000（16kHz）
        :param filename: 要保存的WAV文件名
        """
        # 转换音频数据为 numpy 数组（假设是 float32）
        audio_array = np.frombuffer(audio_data, dtype=np.float32)

        # 将 float32 转换为 int16，这是 WAV 文件常见的格式
        # 需要确保音频数据的幅度在 -32768 到 32767 之间
        audio_array_int16 = np.int16(audio_array * 32767)

        # 创建一个 wave 文件
        with wave.open(filename, 'wb') as wav_file:
            # 设置参数
            n_channels = 1  # 单声道
            sampwidth = 2  # 每个采样点2字节（16位）
            n_frames = len(audio_array_int16)

            wav_file.setnchannels(n_channels)  # 设置声道数
            wav_file.setsampwidth(sampwidth)  # 设置采样宽度
            wav_file.setframerate(sample_rate)  # 设置采样率
            wav_file.setnframes(n_frames)  # 设置帧数

            # 将音频数据写入文件
            wav_file.writeframes(audio_array_int16.tobytes())


    with MicrophoneStream() as stream:
        await websocket.send(json.dumps({"event": "start"}))
        logger.info("录音启动...")

        merged_audio = bytearray()
        a = 0
        async for chunk in stream.async_generator():
            merged_audio.extend(chunk)
            # a += 1
            # if a >= 2:
            #     # save_audio_to_wav(merged_audio, 16000, './111.wav')
            #     await websocket.send(json.dumps({'event':'ping'}))
            #     a = 0

            await websocket.send(json.dumps({
                "event": "media",
                "media": {
                    "payload": base64.b64encode(merged_audio).decode(),
                    # "language_code":['zh-Hans-CN', 'mul_translate','en-US']
                    "language_code":['zh-Hans-CN', 'transcript']
                }
            }))
            merged_audio = bytearray()
            # a = 0

        await websocket.send(json.dumps({"event": "stop"}))
        logger.info("录音结束...")

async def main():
    """主连接逻辑"""
    retries = 0
    while retries < MAX_RETRIES:
        try:
            async with websockets.connect(URI,ping_interval=30, ping_timeout=60) as ws:
                logger.success("连接成功")
                await asyncio.gather(
                    send_audio_chunks(ws),
                    receive_responses(ws)
                )
                return  # 成功完成
        except Exception as e:
            retries += 1
            logger.error(f"连接失败 ({retries}/{MAX_RETRIES}): {str(e)}")
            await asyncio.sleep(2)

    logger.error("超过最大重试次数")


if __name__ == "__main__":
    asyncio.run(main())