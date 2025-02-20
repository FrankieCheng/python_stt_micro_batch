import time
import os
import asyncio
import base64
import websockets
import json
from loguru import logger
from vad import (VADIterator, read_audio)

# 音频参数
SAMPLING_RATE = int(os.getenv('SAMPLING_RATE', 16000))
CHUNK = int(os.getenv('CHUNK', SAMPLING_RATE // 2))  # 确保为整数
TIMEOUT = 1000
chunk_size = 2000
language_code = 'zh-Hans-CN'
# URI = os.getenv('SERVER_URI', "ws://ec2-122-248-254-86.ap-southeast-1.compute.amazonaws.com:9080/chat")
URI = os.getenv('SERVER_URI', "ws://34.123.219.129:9081/chat") # 美国
MAX_RETRIES = 3
filename = "/Users/wangjie/Desktop/test_xudan2.wav"

def tensor_to_bytes(tensor):
  # Convert to NumPy (assuming 16-bit PCM audio)
    audio_np = tensor.numpy()

# Convert to bytes (little-endian for WAV format)
    byte_data = audio_np.tobytes()  # Or audio_np.tobytes(order='C') for big-endian
    return byte_data

def generate_chunks(filename, grpc_on=False, chunk_size=chunk_size, language_code=language_code):
    if not filename.endswith('.wav'):
        return

    all_file_chunks = read_audio(filename, sampling_rate=SAMPLING_RATE)

    for index_value in range(0, len(all_file_chunks), SAMPLING_RATE):
        chunk = all_file_chunks[index_value:index_value + SAMPLING_RATE]
        if len(chunk) == 0:
            break

        if grpc_on:
            yield tensor_to_bytes(chunk)
        else:
            yield chunk

        if len(chunk) < SAMPLING_RATE:
            break

        time.sleep(chunk_size / (SAMPLING_RATE - 1))

async def send_audio_chunks(websocket):
    await websocket.send(json.dumps({"event": "start"}))
    logger.info("处理文件数据...")

    for chunk in generate_chunks(filename,grpc_on=True):
        encoded = base64.b64encode(chunk).decode()
        await websocket.send(json.dumps({
            "event": "media",
            "media": {
                "payload": encoded,
                "language_code":['zh-Hans-CN', 'mul_translate','en-US']
            }
        }))

    await websocket.send(json.dumps({"event": "stop"}))
    logger.info("文件数据处理完成...")

async def receive_responses(websocket):
    try:
        async for message in websocket:
            try:
                data = json.loads(message)
                if "transcript" in data and data["transcript"] is not None:
                    logger.info(f"实时转录: {data['transcript']}")
                elif "error" in data:
                    logger.error(f"服务端错误: {data['error']}")
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                logger.error(f"无效的响应: {e}")
    except websockets.exceptions.ConnectionClosed:
        logger.warning("服务端连接已关闭")

async def connect_with_retries():
    retries = 0
    while retries < MAX_RETRIES:
        try:
            async with websockets.connect(URI) as ws:
                logger.success("连接成功")
                return ws
        except Exception as e:
            retries += 1
            logger.error(f"连接失败 ({retries}/{MAX_RETRIES}): {str(e)}")
            await asyncio.sleep(2)
    raise ConnectionError("超过最大重试次数")

async def main():
    retries = 0
    while retries < MAX_RETRIES:
        try:
            async with websockets.connect(URI) as ws:
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