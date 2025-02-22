import asyncio
import threading

import psutil
import torch
import websockets
import base64
import json
import numpy as np
from loguru import logger
# from ddd import TranscriptionServer
from translate_server import TranscriptionServer
import os
import wave


# PROJECT_ID = os.getenv('PROJECT_ID', "cloudplus1-test-new")
# LOCATION = os.getenv('LOCATION', "us-central1")
PROJECT_ID = os.getenv('PROJECT_ID', "zte-gemini")
LOCATION = os.getenv('LOCATION', "us-central1")
RECOGNIZER = os.getenv('RECOGNIZER', "-")
HTTP_SERVER_PORT = int(os.getenv('HTTP_SERVER_PORT', 9082))
SAMPLING_RATE = 16000  # 假设采样率为 16kHz



# 处理 WebSocket 连接的函数
async def handle_connection(websocket):
    async with TranscriptionServer(PROJECT_ID, LOCATION, RECOGNIZER, websocket) as transcript_server:
        try:
            await handle_socket(websocket, transcript_server)
            await transcript_server.clear()
        except Exception as e:
            logger.error(f"Connection error: {e}")
            raise





async def log_memory():
    process = psutil.Process(os.getpid())
    logger.info(f"Memory used: {process.memory_info().rss / 1024 ** 2:.2f} MB")


async def handle_socket(websocket,transcript_server):
    try:
        start_task = asyncio.create_task(transcript_server.start())
        async for message in websocket:
            await log_memory()

            try:
                data = json.loads(message)
                # 处理批量消息
                if isinstance(data, list):
                    await process_single_message(websocket, data, transcript_server)
                else:
                    await process_single_message(websocket, [data],transcript_server)
            except json.JSONDecodeError as e:
                logger.error(f"Failed to decode JSON: {e}")
                await websocket.send(json.dumps({"error": "Invalid JSON format"}))
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                await websocket.send(json.dumps({"error": "Unexpected error occurred"}))

    except websockets.exceptions.ConnectionClosedOK:
        logger.info("Connection closed normally")
    except websockets.exceptions.ConnectionClosedError as e:
        logger.error(f"Connection closed with error: {e}")
    finally:
        start_task.cancel()  # 确保任务退出
        await start_task  # 等待任务清理
        pass


async def process_single_message(websocket, data,transcript_server):
    try:
        for single_msg in data:
            event = single_msg["event"]
            if event == "start":
                logger.debug("Audio stream started")
            elif event == "media":
                await transcript_server.add_request(single_msg)
            elif event == "stop":
                logger.debug("Audio stream stopped")
            elif event == "ping":
                # 处理心跳消息
                await websocket.send(json.dumps({"event": "pong"}))
    except Exception as e:
        logger.error(e)
        await websocket.send(json.dumps({"error": f"{e}"}))



# 启动 WebSocket 服务器的异步函数
async def start_server():
    server = await websockets.serve(handle_connection, "0.0.0.0", HTTP_SERVER_PORT, ping_interval = 30, ping_timeout = 60, max_size = 1024*1024*10)
    logger.success(f"Server listening on: ws://localhost:{HTTP_SERVER_PORT}/chat")
    await server.wait_closed()

if __name__ == "__main__":
    asyncio.run(start_server())
