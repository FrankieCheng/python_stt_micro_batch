import asyncio
import grpc
import os
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from dotenv import load_dotenv

# Import your generated gRPC files
import stt_pb2 as stt__pb2
import stt_pb2_grpc as stt__pb2__grpc

load_dotenv() # Load environment variables if you use a .env file

# --- Configuration ---
# Get gRPC server address from environment variable or use default
GRPC_SERVER_ADDRESS = os.getenv("GRPC_SERVER_ADDRESS", "localhost:8000")
# Timeout for the *entire* gRPC stream (adjust as needed)
_TIMEOUT_SECONDS_STREAM = 1000

# --- FastAPI Setup ---
app = FastAPI()
templates = Jinja2Templates(directory="templates")

# --- gRPC Client Setup (Shared Channel) ---
# Create a single channel to be reused - handle potential connection errors in practice
try:
    grpc_channel = grpc.aio.insecure_channel(GRPC_SERVER_ADDRESS)
    # You might want to add readiness checks or connection retries here
    print(f"Attempting to connect to gRPC server at {GRPC_SERVER_ADDRESS}...")
    # Add a quick check if the channel is ready (optional, requires grpcio-health-checking)
    # Or just proceed and handle errors during the call
except Exception as e:
    print(f"Failed to create gRPC channel: {e}")
    grpc_channel = None # Indicate failure


# --- Helper Function (from original script, adapted) ---
def build_request_body(chunk, language_code):
    # IMPORTANT: Ensure the 'chunk' bytes received from the browser
    # are in the RAW format (e.g., PCM 16-bit mono 16kHz) expected
    # by your specific gRPC server's AudioRequest.
    # Decoding/resampling might be needed here!
    return stt__pb2.SpeechChunkRequest(
        content=stt__pb2.AudioRequest(audio=chunk),
        config=stt__pb2.StreamingRecognizeRequest(
            streaming_config=stt__pb2.StreamingConfig(
                config=stt__pb2.RecognitionConfig(
                    language_codes=[language_code]
                )
            )
        )
    )

# --- WebSocket Endpoint ---
@app.websocket("/ws/stt/{language_code}")
async def websocket_endpoint(websocket: WebSocket, language_code: str):
    await websocket.accept()
    print(f"WebSocket connection established for language: {language_code}")

    if not grpc_channel:
        print("gRPC channel not available.")
        await websocket.close(code=1011, reason="Backend gRPC connection failed")
        return

    # Queues to bridge WebSocket incoming audio and gRPC outgoing stream
    grpc_request_queue = asyncio.Queue()
    grpc_response_queue = asyncio.Queue()

    stt_service_stub = stt__pb2__grpc.ListenerStub(grpc_channel)

    # --- gRPC Streaming Coroutines ---
    async def grpc_request_generator():
        """Async generator for gRPC requests, reads from queue."""
        while True:
            chunk = await grpc_request_queue.get()
            if chunk is None: # Signal to end stream
                 # print("gRPC request generator received None, stopping.")
                 break
            # print(f"Sending chunk of size {len(chunk)} to gRPC")
            yield build_request_body(chunk=chunk, language_code=language_code)
            await asyncio.sleep(0.01) # Small sleep to prevent tight loop if queue fills fast


    async def grpc_response_processor():
        """Processes responses from gRPC and puts them on response queue."""
        try:
            # print("Starting gRPC DoSpeechToText call...")
            async for response in stt_service_stub.DoSpeechToText(
                grpc_request_generator(), timeout=_TIMEOUT_SECONDS_STREAM
            ):
                # print(f"Received gRPC response: {response.results}")
                await grpc_response_queue.put(response)
            await grpc_response_queue.put(None) # Signal end of responses
            # print("gRPC response stream finished.")
        except grpc.aio.AioRpcError as e:
            print(f"gRPC Error during DoSpeechToText: {e.code()} - {e.details()}")
            await grpc_response_queue.put(None) # Signal error/end
        except Exception as e:
            print(f"Unexpected error in gRPC response processor: {e}")
            await grpc_response_queue.put(None) # Signal error/end


    async def send_results_to_client():
        """Sends results from response queue back to WebSocket client."""
        try:
            while True:
                response = await grpc_response_queue.get()
                if response is None:
                    # print("Result sender received None, stopping.")
                    break

                if response.results:
                    result = response.results[0]
                    if result.alternatives:
                        transcript = result.alternatives[0].transcript
                        is_final = result.is_final
                        # print(f"Sending to client: {'FINAL' if is_final else 'INTERIM'}: {transcript}")
                        await websocket.send_json({
                            "transcript": transcript,
                            "is_final": is_final
                        })
                await asyncio.sleep(0.01) # Prevent tight loop
        except WebSocketDisconnect:
            print("Client disconnected while sending results.")
        except Exception as e:
            print(f"Error sending results to client: {e}")


    # --- Start gRPC tasks ---
    response_task = asyncio.create_task(grpc_response_processor())
    send_task = asyncio.create_task(send_results_to_client())

    # --- Main Loop: Receive audio from WebSocket, put on queue ---
    try:
        while True:
            # Receive raw audio bytes from the client
            data = await websocket.receive_bytes()
            # print(f"Received {len(data)} bytes from WebSocket.")
            # Put the raw bytes onto the queue for gRPC processing
            # *** This assumes the browser sends data in the format gRPC expects! ***
            await grpc_request_queue.put(data)

    except WebSocketDisconnect:
        print("WebSocket client disconnected.")
    except Exception as e:
        print(f"An error occurred in the WebSocket receive loop: {e}")
    finally:
        print("Cleaning up WebSocket connection...")
        # Signal gRPC request generator to stop
        await grpc_request_queue.put(None)

        # Wait briefly for tasks to potentially finish processing
        await asyncio.sleep(0.5)

        # Cancel pending tasks if they haven't finished
        if response_task and not response_task.done():
            response_task.cancel()
            print("Cancelled gRPC response task.")
        if send_task and not send_task.done():
            send_task.cancel()
            print("Cancelled client send task.")

        # Wait for cancellation (optional but good practice)
        try:
            await asyncio.gather(response_task, send_task, return_exceptions=True)
        except asyncio.CancelledError:
            print("Tasks successfully cancelled.")

        print("WebSocket cleanup complete.")


# --- HTML Frontend Route ---
@app.get("/", response_class=HTMLResponse)
async def get(request: Request):
    # Pass language codes or other config to template if needed
    return templates.TemplateResponse("index.html", {"request": request})

# --- Optional: Add endpoint to list languages or get config ---
# @app.get("/config")
# async def get_config():
#     # Return supported languages, etc.
#     return {"grpc_server": GRPC_SERVER_ADDRESS, "supported_languages": ["zh-Hans-CN", "en-US"]}


if __name__ == "__main__":
    import uvicorn
    print("Starting FastAPI server...")
    # Run with: uvicorn main:app --host 0.0.0.0 --port 8000 --reload
    # Change host/port as needed
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)