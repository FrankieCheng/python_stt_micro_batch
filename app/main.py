# app/main.py
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse 
import logging
import os

# Ensure stt_pb2.py and stt_pb2_grpc.py are discoverable
# This might require adjusting PYTHONPATH or your project structure
from . import grpc_client_manager # Assuming grpc_client_manager is in the same directory

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(asctime)s: %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI()

# Determine the correct path to the static directory
# This assumes 'static' is a subdirectory of 'app' and 'main.py' is in 'app'
static_dir = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/")
async def get_root():
    html_file_path = os.path.join(static_dir, "index.html")
    return FileResponse(html_file_path)

class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket {websocket.client} connected.")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"WebSocket {websocket.client} disconnected.")

    async def send_json_to_websocket(self, data: dict, websocket: WebSocket):
        try:
            await websocket.send_json(data)
        except WebSocketDisconnect:
            logger.warning(f"Attempted to send to a disconnected websocket: {websocket.client}")
            self.disconnect(websocket) # Ensure cleanup
        except Exception as e:
            logger.error(f"Error sending JSON to websocket {websocket.client}: {e}")
            self.disconnect(websocket)


manager = ConnectionManager()

@app.websocket("/ws/stt")
async def websocket_stt_endpoint(
    websocket: WebSocket,
    language: str = Query("en-US", description="Language code for STT (e.g., en-US, zh-Hans-CN)")
):
    await manager.connect(websocket)

    async def audio_chunk_receiver():
        """Receives audio chunks from WebSocket and yields them."""
        try:
            while True:
                # Browser client (script.js) should send raw audio bytes (ArrayBuffer)
                data = await websocket.receive_bytes()
                if not data: # Should not happen with bytes, but good for robustness
                    logger.info("Received empty data, treating as end of stream for this client.")
                    yield None # Signal end of stream
                    break
                # logger.debug(f"Received audio chunk of size: {len(data)} bytes from {websocket.client}")
                yield data
        except WebSocketDisconnect:
            logger.info(f"WebSocket {websocket.client} disconnected while receiving audio.")
            yield None # Signal end of stream
        except Exception as e:
            logger.error(f"Error receiving audio from WebSocket {websocket.client}: {e}", exc_info=True)
            yield None # Signal end of stream


    try:
        audio_iterator = audio_chunk_receiver()
        async for result_data in grpc_client_manager.stream_audio_to_grpc(audio_iterator, language):
            await manager.send_json_to_websocket(result_data, websocket)
    except Exception as e:
        logger.error(f"Overall error in WebSocket handler for {websocket.client}: {e}", exc_info=True)
        # Try to send an error message to the client if the websocket is still somewhat alive
        error_payload = {"type": "error", "message": f"An internal server error occurred: {str(e)}"}
        await manager.send_json_to_websocket(error_payload, websocket)
    finally:
        manager.disconnect(websocket)

if __name__ == "__main__":
    # This is for running with `python app/main.py`
    # For production, use: uvicorn app.main:app --host 0.0.0.0 --port 8000
    import uvicorn
    logger.info("Starting FastAPI server with Uvicorn on http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)