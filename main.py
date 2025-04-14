import asyncio
import logging
import grpc
import ffmpeg # Added for audio decoding
import numpy as np # Added for potential byte manipulation (though less needed now)
import subprocess # To manage ffmpeg process

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse # To serve the HTML page directly

# Import the generated gRPC files (assuming they are in the same directory or PYTHONPATH)
# You might need to adjust the import path based on your project structure
try:
    import stt_pb2
    import stt_pb2_grpc
    # Import necessary types for state checking if running directly
    from starlette.websockets import WebSocketState
except ImportError:
    print("Error: Could not import stt_pb2 and stt_pb2_grpc.")
    print("Ensure the generated Python files from your .proto file are accessible.")
    # You might need to generate these using:
    # python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. stt.proto
    # Replace stt.proto with your actual proto file name.
    exit(1)


# --- Configuration ---
# Ensure these match the address and port your gRPC server is running on
GRPC_SERVER_ADDRESS = "localhost:9080"
# Default language code - consider making this configurable via the frontend
DEFAULT_LANGUAGE_CODE = "en-US" # Example: "en-US", "zh-CN", etc.
# Audio decoding settings (must match server expectation)
TARGET_SAMPLE_RATE = 16000
TARGET_FORMAT = 'f32le' # float32 little-endian PCM
TARGET_CHANNELS = 1
BYTES_PER_SAMPLE = 4 # float32 uses 4 bytes

# --- Buffering Configuration ---
BUFFER_DURATION_SECONDS = 0.6 # Target duration for each chunk sent to gRPC
# Calculate target buffer size in bytes
TARGET_BUFFER_SIZE = int(TARGET_SAMPLE_RATE * TARGET_CHANNELS * BYTES_PER_SAMPLE * BUFFER_DURATION_SECONDS)
# Define a smaller read size from ffmpeg pipe for responsiveness
FFMPEG_READ_SIZE = 4096 # Read in smaller chunks from ffmpeg


# --- Logging ---
FORMAT = '%(levelname)s: %(asctime)s: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT)
logger = logging.getLogger('FastAPI_STT_Client_Buffered')

# --- FastAPI App ---
app = FastAPI()

# --- HTML Content (Served directly for simplicity) ---
# We'll embed the HTML content here. For larger applications,
# you'd typically serve static files separately.
html_content = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Real-time Transcription</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;700&display=swap" rel="stylesheet">
    <style>
        body {
            font-family: 'Inter', sans-serif;
        }
        /* Basic transition for button hover */
        button {
            transition: background-color 0.2s ease-in-out, transform 0.1s ease-in-out;
        }
        button:active {
            transform: scale(0.98);
        }
        /* Style for the recording indicator */
        .recording-indicator {
            width: 12px;
            height: 12px;
            background-color: red;
            border-radius: 50%;
            display: inline-block;
            margin-left: 8px;
            animation: pulse 1.5s infinite ease-in-out;
        }
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }
    </style>
</head>
<body class="bg-gray-100 flex items-center justify-center min-h-screen">
    <div class="bg-white p-8 rounded-lg shadow-lg w-full max-w-2xl">
        <h1 class="text-2xl font-bold mb-6 text-center text-gray-800">Real-time Speech-to-Text</h1>

        <div class="flex justify-center space-x-4 mb-6">
            <button id="startButton" class="bg-green-500 hover:bg-green-600 text-white font-bold py-2 px-6 rounded-lg shadow focus:outline-none focus:ring-2 focus:ring-green-400 focus:ring-opacity-75">
                Start Recording
            </button>
            <button id="stopButton" class="bg-red-500 hover:bg-red-600 text-white font-bold py-2 px-6 rounded-lg shadow focus:outline-none focus:ring-2 focus:ring-red-400 focus:ring-opacity-75" disabled>
                Stop Recording <span id="indicator" class="hidden recording-indicator"></span>
            </button>
        </div>

        <div class="mb-4">
            <label for="status" class="block text-sm font-medium text-gray-700">Status:</label>
            <p id="status" class="mt-1 text-sm text-gray-600 bg-gray-50 p-3 rounded-md border border-gray-200">Not connected</p>
        </div>

        <div>
            <label for="transcript" class="block text-sm font-medium text-gray-700">Transcript:</label>
            <div id="transcript" class="mt-1 h-60 overflow-y-auto p-4 border border-gray-300 rounded-md bg-white shadow-sm">
                <p class="text-gray-500 italic">Waiting for transcription...</p>
            </div>
        </div>
            <p id="error" class="mt-4 text-sm text-red-600"></p>
    </div>

    <script>
        const startButton = document.getElementById('startButton');
        const stopButton = document.getElementById('stopButton');
        const transcriptDiv = document.getElementById('transcript');
        const statusDiv = document.getElementById('status');
        const errorDiv = document.getElementById('error');
        const indicator = document.getElementById('indicator');

        let websocket;
        let mediaRecorder;
        let audioContext;
        let audioStream;
        let processor; // ScriptProcessorNode for audio processing

        const WS_URL = `ws://${window.location.host}/ws`; // Use relative host

        function updateStatus(message, isError = false) {
            console.log("Status:", message);
            statusDiv.textContent = message;
            statusDiv.className = `mt-1 text-sm p-3 rounded-md border ${isError ? 'text-red-700 bg-red-50 border-red-200' : 'text-gray-600 bg-gray-50 border-gray-200'}`;
        }

        function updateTranscript(newText) {
            // Clear placeholder if it exists
            const placeholder = transcriptDiv.querySelector('.italic');
            if (placeholder) {
                placeholder.remove();
            }
                // Append new text and scroll down
            const p = document.createElement('p');
            p.textContent = newText;
            transcriptDiv.appendChild(p);
            transcriptDiv.scrollTop = transcriptDiv.scrollHeight; // Auto-scroll
        }

        function clearError() {
            errorDiv.textContent = '';
        }
            function displayError(message) {
            console.error("Error:", message);
            errorDiv.textContent = `Error: ${message}`;
            updateStatus(`Error: ${message}`, true);
        }


        async function connectWebSocket() {
            return new Promise((resolve, reject) => {
                if (websocket && websocket.readyState === WebSocket.OPEN) {
                    resolve(websocket);
                    return;
                }

                updateStatus("Connecting to server...");
                websocket = new WebSocket(WS_URL);

                websocket.onopen = () => {
                    updateStatus("Connected. Ready to record.");
                    clearError();
                    resolve(websocket);
                };

                websocket.onmessage = (event) => {
                    console.log("Received message:", event.data);
                    // Assuming the server sends the final transcription text directly
                        if (typeof event.data === 'string') {
                        updateTranscript(event.data);
                    } else {
                            console.warn("Received non-string message:", event.data);
                    }
                };

                websocket.onerror = (event) => {
                    console.error("WebSocket error:", event);
                    displayError("WebSocket connection error. Is the backend server running?");
                    reject(new Error("WebSocket error"));
                };

                websocket.onclose = (event) => {
                    updateStatus(`Disconnected: ${event.reason || 'Connection closed'}`);
                    console.log("WebSocket closed:", event.code, event.reason);
                    websocket = null;
                    stopRecording(); // Ensure recording stops if connection drops
                    reject(new Error(`WebSocket closed: ${event.code}`));
                };
            });
        }

        async function startRecording() {
            clearError();
            if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
                displayError("getUserMedia not supported on your browser!");
                return;
            }

            try {
                await connectWebSocket(); // Ensure connection before starting
                if (!websocket || websocket.readyState !== WebSocket.OPEN) {
                        displayError("WebSocket connection not established.");
                    return;
                }

                updateStatus("Requesting microphone access...");
                audioStream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });

                updateStatus("Microphone access granted. Starting recording...");

                // --- Audio Processing Setup ---
                // Use AudioContext for more control if needed, or directly MediaRecorder
                audioContext = new (window.AudioContext || window.webkitAudioContext)();
                const source = audioContext.createMediaStreamSource(audioStream);

                // --- MediaRecorder Setup ---
                // We will send compressed audio (e.g., Opus/WebM) to the backend,
                // which will handle decoding it to raw PCM for the gRPC server.
                const options = { mimeType: 'audio/webm;codecs=opus' }; // Or try 'audio/ogg;codecs=opus'
                try {
                        mediaRecorder = new MediaRecorder(audioStream, options);
                } catch (e) {
                    console.warn(`MimeType ${options.mimeType} not supported, trying default. Error: ${e}`);
                        try {
                            mediaRecorder = new MediaRecorder(audioStream); // Try default
                        } catch (e2) {
                        displayError(`Could not create MediaRecorder: ${e2}. Your browser might not support MediaRecorder or required codecs.`);
                        if (audioContext && audioContext.state !== 'closed') audioContext.close(); // Clean up context
                        return;
                        }
                }
                console.log("Using MediaRecorder with options:", mediaRecorder.mimeType);

                mediaRecorder.ondataavailable = (event) => {
                    if (event.data.size > 0 && websocket && websocket.readyState === WebSocket.OPEN) {
                        // console.log("Sending audio chunk, size:", event.data.size);
                        websocket.send(event.data); // Send Blob directly
                    }
                };

                mediaRecorder.onstart = () => {
                    updateStatus("Recording...");
                    startButton.disabled = true;
                    stopButton.disabled = false;
                    indicator.classList.remove('hidden');
                    // Clear previous transcript
                    transcriptDiv.innerHTML = '<p class="text-gray-500 italic">Listening...</p>';
                };

                mediaRecorder.onstop = () => {
                    // Don't change status if already disconnected or showing error
                    if (websocket && websocket.readyState === WebSocket.OPEN) {
                            updateStatus("Recording stopped. Processing final audio...");
                    }
                    startButton.disabled = false;
                    stopButton.disabled = true;
                    indicator.classList.add('hidden');
                        // Clean up audio resources
                    if (audioStream) {
                        audioStream.getTracks().forEach(track => track.stop());
                    }
                    if (audioContext && audioContext.state !== 'closed') {
                        audioContext.close();
                    }
                    audioStream = null;
                    audioContext = null;
                    mediaRecorder = null;
                };

                mediaRecorder.onerror = (event) => {
                    // Use event.error for more details if available
                    const errorMsg = event.error ? event.error.message : 'Unknown MediaRecorder error';
                    displayError(`MediaRecorder error: ${errorMsg}`);
                    stopRecording(); // Stop on error
                };

                // Start recording and send chunks periodically
                // The interval here determines how often the browser sends data *to the backend*.
                // It doesn't directly control the chunk size sent *to the gRPC server* anymore.
                // A smaller interval (e.g., 100-200ms) might be better for lower latency perception,
                // even though the backend buffers it. Let's try 200ms.
                mediaRecorder.start(200); // Send data every 200ms to backend

            } catch (err) {
                console.error("Error starting recording:", err);
                if (err.name === "NotAllowedError" || err.name === "PermissionDeniedError") {
                        displayError("Microphone permission denied. Please allow access in your browser settings.");
                } else if (err.name === "NotFoundError" || err.name === "DevicesNotFoundError") {
                        displayError("No microphone found.");
                } else {
                    displayError(`Could not start recording: ${err.message}`);
                }
                // Clean up potential partial setup
                if (websocket && websocket.readyState === WebSocket.OPEN) {
                    websocket.close();
                }
                stopRecording(); // Ensure UI reset
            }
        }

        function stopRecording() {
            if (mediaRecorder && mediaRecorder.state === "recording") {
                mediaRecorder.stop(); // This will trigger onstop event
            } else {
                // Manual cleanup if recorder wasn't fully started or already stopped
                if (audioStream) {
                    audioStream.getTracks().forEach(track => track.stop());
                }
                    if (audioContext && audioContext.state !== 'closed') {
                    audioContext.close();
                }
                startButton.disabled = false;
                stopButton.disabled = true;
                indicator.classList.add('hidden');
                audioStream = null;
                audioContext = null;
                mediaRecorder = null;
                    // Don't change status if it's already showing an error or disconnected
                    if (!statusDiv.textContent.toLowerCase().includes('error') && !statusDiv.textContent.toLowerCase().includes('disconnect')) {
                    updateStatus("Recording stopped.");
                    }
            }
                // Optionally close WebSocket when stopping recording, or keep it open
                // if (websocket && websocket.readyState === WebSocket.OPEN) {
                //     websocket.close();
                // }
        }

        startButton.addEventListener('click', startRecording);
        stopButton.addEventListener('click', stopRecording);

        // Initial status
        updateStatus("Ready. Click 'Start Recording'.");

        // Graceful shutdown
        window.addEventListener('beforeunload', () => {
            if (websocket && websocket.readyState === WebSocket.OPEN) {
                websocket.close();
            }
            stopRecording(); // Stop mic access
        });

    </script>
</body>
</html>
"""

# --- Serve HTML Page ---
@app.get("/", response_class=HTMLResponse)
async def get_html():
    return HTMLResponse(content=html_content)


# --- WebSocket Endpoint ---
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logger.info(f"WebSocket connection accepted from {websocket.client.host}:{websocket.client.port}")
    logger.info(f"Audio buffer target size: {TARGET_BUFFER_SIZE} bytes (~{BUFFER_DURATION_SECONDS} seconds)")

    grpc_channel = None
    grpc_stub = None
    response_stream = None
    grpc_send_task = None
    grpc_receive_task = None
    ffmpeg_process = None
    feed_ffmpeg_task = None
    read_ffmpeg_task = None

    # Queue to pass DECODED and BUFFERED audio from ffmpeg task to gRPC task
    audio_queue = asyncio.Queue()

    try:
        # --- Start FFmpeg subprocess for decoding ---
        logger.info("Starting ffmpeg process for audio decoding...")
        try:
            # Input format might need adjustment based on what MediaRecorder sends
            # Common options: 'webm', 'ogg'. FFmpeg is often good at auto-detecting.
            ffmpeg_command = (
                ffmpeg
                .input('pipe:0') # Let ffmpeg detect container format
                .output('pipe:1', format=TARGET_FORMAT, acodec='pcm_f32le', ac=TARGET_CHANNELS, ar=TARGET_SAMPLE_RATE)
                .global_args('-hide_banner', '-loglevel', 'warning') # Show warnings/errors
                .compile()
            )
            # Use asyncio's subprocess management
            ffmpeg_process = await asyncio.create_subprocess_exec(
                *ffmpeg_command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE # Capture stderr for debugging
            )
            logger.info(f"ffmpeg process started with PID: {ffmpeg_process.pid}")

        except FileNotFoundError:
            logger.error("ffmpeg command not found. Please ensure ffmpeg is installed and in your PATH.")
            await websocket.close(code=1011, reason="Internal server error: ffmpeg not found")
            return
        except Exception as e:
            logger.error(f"Failed to start ffmpeg process: {e}", exc_info=True)
            await websocket.close(code=1011, reason="Internal server error: failed to start decoder")
            return

        # --- Establish gRPC connection ---
        logger.info(f"Connecting to gRPC server at {GRPC_SERVER_ADDRESS}")
        # Add options for keepalive, etc., if needed for long connections
        grpc_options = [
            ('grpc.keepalive_time_ms', 30000), # Send keepalive ping every 30s
            ('grpc.keepalive_timeout_ms', 10000), # Wait 10s for pong response
            ('grpc.keepalive_permit_without_calls', True), # Allow keepalive pings when there are no calls
            ('grpc.http2.min_ping_interval_without_data_ms', 15000), # Allow pings more often when idle
            ('grpc.http2.max_pings_without_data', 0), # Allow unlimited pings without data
        ]
        grpc_channel = grpc.aio.insecure_channel(GRPC_SERVER_ADDRESS, options=grpc_options)
        try:
            # Increased timeout for potentially slower networks or server startup
            await asyncio.wait_for(grpc_channel.channel_ready(), timeout=10.0)
            logger.info("gRPC channel ready.")
        except asyncio.TimeoutError:
            logger.error("gRPC connection timed out.")
            raise # Let outer try/except handle closing
        except grpc.aio.AioRpcError as e:
                logger.error(f"gRPC channel connection failed: {e}")
                raise # Let outer try/except handle closing

        grpc_stub = stt_pb2_grpc.ListenerStub(grpc_channel)

        # --- gRPC Streaming Call Setup ---
        # This function generates requests (now containing buffered raw PCM) to send to the gRPC server
        async def request_generator():
            is_first_request = True
            try:
                while True:
                    # Get the buffered chunk (approx TARGET_BUFFER_SIZE bytes)
                    buffered_chunk = await audio_queue.get()
                    if buffered_chunk is None:
                        logger.info("Audio queue signaled stop. Ending gRPC request stream.")
                        break
                    if not isinstance(buffered_chunk, bytes) or len(buffered_chunk) == 0:
                            logger.warning(f"Received invalid item from audio queue: type={type(buffered_chunk)}, len={len(buffered_chunk)}. Skipping.")
                            audio_queue.task_done()
                            continue

                    # Construct the request message based on your stt.proto definition
                    try:
                        # Send config only on the first request
                        if is_first_request:
                            recognition_config = stt_pb2.RecognitionConfig(language_codes=[DEFAULT_LANGUAGE_CODE])
                            streaming_config = stt_pb2.StreamingConfig(config=recognition_config)
                            streaming_recognize_request = stt_pb2.StreamingRecognizeRequest(streaming_config=streaming_config)
                            audio_request = stt_pb2.AudioRequest(audio=buffered_chunk) # Use buffered chunk
                            request = stt_pb2.SpeechChunkRequest(
                                content=audio_request,
                                config=streaming_recognize_request
                            )
                            is_first_request = False
                            logger.debug(f"Sending FIRST gRPC request with config and audio chunk size: {len(buffered_chunk)}")
                        else:
                            # Subsequent requests only contain audio
                            audio_request = stt_pb2.AudioRequest(audio=buffered_chunk) # Use buffered chunk
                            request = stt_pb2.SpeechChunkRequest(content=audio_request)
                            logger.debug(f"Sending gRPC request with audio chunk size: {len(buffered_chunk)}")

                        yield request
                    except Exception as e:
                            logger.error(f"Error constructing gRPC request: {e}", exc_info=True)
                            # Don't stop the whole generator for one bad construction
                    finally:
                            audio_queue.task_done() # Notify queue that item is processed
            except asyncio.CancelledError:
                    logger.info("gRPC request_generator cancelled.")
            except Exception as e:
                logger.error(f"Error in gRPC request_generator: {e}", exc_info=True)

        # Start the bidirectional gRPC stream
        logger.info("Calling DoSpeechToText...")
        response_stream = grpc_stub.DoSpeechToText(request_generator())

        # --- Task to receive from gRPC and send to WebSocket ---
        async def receive_grpc_messages():
            # (This function remains largely the same as before)
            try:
                async for response in response_stream:
                    if response and hasattr(response, 'results') and response.results:
                            transcript = ""
                            for result in response.results:
                                if hasattr(result, 'alternatives') and result.alternatives:
                                    if hasattr(result.alternatives[0], 'transcript'):
                                        transcript += result.alternatives[0].transcript + " "
                            if transcript:
                                final_transcript = transcript.strip()
                                logger.info(f"Received transcript: '{final_transcript}'")
                                if websocket.client_state == WebSocketState.CONNECTED:
                                    await websocket.send_text(final_transcript)
                                else:
                                    logger.warning("WebSocket no longer connected, cannot send transcript.")
                                    break # Stop trying if websocket is closed
                    else:
                        logger.debug(f"Received gRPC response with no processable transcript: {response}")
            except grpc.aio.AioRpcError as e:
                # Log specific gRPC error codes
                log_level = logging.ERROR if e.code() != grpc.StatusCode.CANCELLED else logging.INFO
                logger.log(log_level, f"gRPC receive error: Code={e.code()} Details='{e.details()}'")
                if websocket.client_state == WebSocketState.CONNECTED:
                    # Avoid closing if the error is just cancellation
                    if e.code() != grpc.StatusCode.CANCELLED:
                        await websocket.close(code=1011, reason=f"STT Service Error: {e.details() or e.code().name}")
            except asyncio.CancelledError:
                    logger.info("gRPC receive task cancelled.")
            except Exception as e:
                logger.error(f"Error receiving from gRPC: {e}", exc_info=True)
                if websocket.client_state == WebSocketState.CONNECTED:
                    await websocket.close(code=1011, reason="Internal server error during transcription")
            finally:
                logger.info("gRPC receive task finished.")


        # --- Task to receive from WebSocket and feed into ffmpeg stdin ---
        async def feed_ffmpeg():
            bytes_fed = 0
            try:
                while True:
                    # Receive compressed audio data as bytes from the frontend
                    audio_chunk = await websocket.receive_bytes()
                    bytes_fed += len(audio_chunk)
                    # logger.debug(f"Received WebSocket audio chunk size: {len(audio_chunk)}")
                    if ffmpeg_process and ffmpeg_process.stdin and not ffmpeg_process.stdin.is_closing():
                        try:
                            ffmpeg_process.stdin.write(audio_chunk)
                            await ffmpeg_process.stdin.drain()
                            # logger.debug(f"Wrote {len(audio_chunk)} bytes to ffmpeg stdin")
                        except (ConnectionResetError, BrokenPipeError) as e:
                            logger.warning(f"ffmpeg stdin pipe closed unexpectedly: {e}. Stopping feed.")
                            break # Stop feeding if pipe is broken
                        except Exception as e:
                            logger.error(f"Error writing to ffmpeg stdin: {e}", exc_info=True)
                            break # Stop on other write errors
                    else:
                        logger.warning("ffmpeg stdin is not available or closing, stopping feed.")
                        break
            except WebSocketDisconnect as e:
                logger.info(f"WebSocket disconnected by client (code: {e.code}, reason: {e.reason}). Stopping ffmpeg feed.")
            except asyncio.CancelledError:
                logger.info("feed_ffmpeg task cancelled.")
            except Exception as e:
                    logger.error(f"Error receiving from WebSocket or feeding ffmpeg: {e}", exc_info=True)
            finally:
                # Signal ffmpeg that no more data is coming
                if ffmpeg_process and ffmpeg_process.stdin and not ffmpeg_process.stdin.is_closing():
                    try:
                        logger.info(f"Closing ffmpeg stdin after feeding {bytes_fed} bytes.")
                        ffmpeg_process.stdin.close()
                        await ffmpeg_process.stdin.wait_closed() # Ensure it's closed
                    except Exception as e:
                        logger.error(f"Error closing ffmpeg stdin: {e}", exc_info=True)
                logger.info("feed_ffmpeg task finished.")


        # --- Task to read decoded PCM from ffmpeg stdout, buffer, and put onto queue ---
        async def read_ffmpeg_output():
            audio_buffer = bytearray() # Use bytearray for efficient appending
            bytes_read_total = 0
            chunks_sent = 0
            try:
                while ffmpeg_process and ffmpeg_process.stdout and not ffmpeg_process.stdout.at_eof():
                    # Read decoded data in chunks from ffmpeg's output pipe
                    # Use the smaller FFMPEG_READ_SIZE here
                    decoded_chunk = await ffmpeg_process.stdout.read(FFMPEG_READ_SIZE)
                    if not decoded_chunk:
                        logger.info("ffmpeg stdout EOF reached.")
                        break # Exit loop if ffmpeg closes its output

                    bytes_read_total += len(decoded_chunk)
                    audio_buffer.extend(decoded_chunk) # Add new data to buffer

                    # Check if buffer has enough data to send
                    while len(audio_buffer) >= TARGET_BUFFER_SIZE:
                        # Extract the chunk to send
                        chunk_to_send = bytes(audio_buffer[:TARGET_BUFFER_SIZE])
                        # Remove the sent chunk from the buffer
                        audio_buffer = audio_buffer[TARGET_BUFFER_SIZE:]

                        # Put the complete chunk onto the queue for gRPC
                        await audio_queue.put(chunk_to_send)
                        chunks_sent += 1
                        # logger.debug(f"Put buffered chunk {chunks_sent} ({len(chunk_to_send)} bytes) onto queue. Buffer remaining: {len(audio_buffer)} bytes.")

                # After EOF, check if there's any remaining data in the buffer
                if len(audio_buffer) > 0:
                    logger.info(f"Sending remaining {len(audio_buffer)} bytes from audio buffer before stopping.")
                    await audio_queue.put(bytes(audio_buffer))
                    chunks_sent += 1

            except asyncio.CancelledError:
                logger.info("read_ffmpeg_output task cancelled.")
            except Exception as e:
                logger.error(f"Error reading from ffmpeg stdout or buffering: {e}", exc_info=True)
            finally:
                # Signal the request_generator to stop by putting None
                logger.info(f"Putting None onto audio queue. Total decoded bytes read: {bytes_read_total}. Total chunks sent: {chunks_sent}")
                await audio_queue.put(None)
                logger.info("read_ffmpeg_output task finished.")


        # --- Run all tasks concurrently ---
        logger.info("Starting concurrent tasks: gRPC receiver, ffmpeg feeder, ffmpeg reader")
        grpc_receive_task = asyncio.create_task(receive_grpc_messages(), name="gRPC Receiver")
        feed_ffmpeg_task = asyncio.create_task(feed_ffmpeg(), name="FFmpeg Feeder")
        read_ffmpeg_task = asyncio.create_task(read_ffmpeg_output(), name="FFmpeg Reader/Bufferer")

        # Wait for tasks to complete - Changed to wait for ALL to complete naturally or by cancellation
        all_tasks = {grpc_receive_task, feed_ffmpeg_task, read_ffmpeg_task}
        done, pending = await asyncio.wait(
            all_tasks,
            return_when=asyncio.ALL_COMPLETED, # Wait for all, not just the first
        )

        logger.info("All processing tasks have completed.")
        # Check if any task raised an exception
        for task in done:
            try:
                task.result() # Raise exception if task failed
            except asyncio.CancelledError:
                pass # Expected during cleanup
            except Exception as e:
                logger.error(f"Task {task.get_name()} finished with error: {e}", exc_info=True)


    except WebSocketDisconnect as e:
            logger.info(f"WebSocket disconnected (caught in main handler - code: {e.code}, reason: {e.reason}).")
    except grpc.aio.AioRpcError as e:
            log_level = logging.ERROR if e.code() != grpc.StatusCode.CANCELLED else logging.INFO
            logger.log(log_level, f"gRPC Error during setup or connection: Code={e.code()} Details='{e.details()}'")
            if websocket.client_state == WebSocketState.CONNECTED and e.code() != grpc.StatusCode.CANCELLED:
                await websocket.close(code=1011, reason=f"STT Service Connection Error: {e.details() or e.code().name}")
    except Exception as e:
        logger.error(f"An unexpected error occurred in websocket_endpoint: {e}", exc_info=True)
        if websocket.client_state == WebSocketState.CONNECTED:
            await websocket.close(code=1011, reason="Internal Server Error")
    finally:
        logger.info("Initiating cleanup for WebSocket connection...")

        # 1. Cancel all potentially running tasks (robust check)
        running_tasks = {grpc_receive_task, feed_ffmpeg_task, read_ffmpeg_task}
        cancelled_tasks = []
        for task in running_tasks:
            if task and not task.done():
                logger.info(f"Cancelling task {task.get_name()}...")
                task.cancel()
                cancelled_tasks.append(task)

        # 2. Wait for explicitly cancelled tasks to finish
        if cancelled_tasks:
            await asyncio.gather(*cancelled_tasks, return_exceptions=True)
            logger.info("Explicitly cancelled asyncio tasks awaited.")

        # 3. Terminate ffmpeg process if it's still running
        if ffmpeg_process and ffmpeg_process.returncode is None:
            logger.info(f"Terminating ffmpeg process (PID: {ffmpeg_process.pid})...")
            try:
                # Try terminate first, then kill if necessary
                ffmpeg_process.terminate()
                try:
                    await asyncio.wait_for(ffmpeg_process.wait(), timeout=2.0)
                    logger.info(f"ffmpeg process terminated gracefully with code: {ffmpeg_process.returncode}")
                except asyncio.TimeoutError:
                    logger.warning(f"ffmpeg process (PID: {ffmpeg_process.pid}) did not terminate gracefully, killing.")
                    ffmpeg_process.kill()
                    await ffmpeg_process.wait()
                    logger.info(f"ffmpeg process killed, return code: {ffmpeg_process.returncode}")
            except ProcessLookupError:
                 logger.warning(f"ffmpeg process (PID: {ffmpeg_process.pid}) already exited.")
            except Exception as e:
                logger.error(f"Error terminating/killing ffmpeg process: {e}", exc_info=True)

        # 4. Log ffmpeg stderr if available (read remaining output)
        if ffmpeg_process and ffmpeg_process.stderr:
                try:
                    # Ensure stderr reading doesn't block indefinitely
                    stderr_output = await asyncio.wait_for(ffmpeg_process.stderr.read(), timeout=1.0)
                    if stderr_output:
                        logger.error(f"ffmpeg stderr output:\n{stderr_output.decode(errors='ignore').strip()}")
                    else:
                        logger.info("No further stderr output from ffmpeg.")
                except asyncio.TimeoutError:
                    logger.warning("Timeout reading remaining ffmpeg stderr.")
                except Exception as e:
                    logger.error(f"Error reading ffmpeg stderr during cleanup: {e}")


        # 5. Close gRPC stream and channel
        # response_stream cancellation is implicitly handled by channel closure or task cancellation
        if grpc_channel:
            logger.info("Closing gRPC channel...")
            await grpc_channel.close(grace=1.0) # Allow 1 second for graceful close
            logger.info("Closed gRPC channel.")

        # 6. Ensure audio queue is drained (optional, helps debugging)
        drained_count = 0
        while not audio_queue.empty():
            try:
                item = audio_queue.get_nowait()
                # logger.debug(f"Draining item from audio queue: type={type(item)}")
                audio_queue.task_done()
                drained_count += 1
            except asyncio.QueueEmpty:
                break
            except Exception as q_exc:
                logger.warning(f"Error draining audio queue: {q_exc}")
        if drained_count > 0:
            logger.info(f"Drained {drained_count} items from audio queue during cleanup.")

        # 7. Ensure WebSocket is closed (FastAPI usually handles this, but explicit check is good)
        if websocket.client_state != WebSocketState.DISCONNECTED:
             logger.warning("WebSocket state was not DISCONNECTED at the end of cleanup. Attempting close.")
             try:
                 await websocket.close(code=1000)
             except Exception as ws_close_err:
                 logger.error(f"Error during final WebSocket close attempt: {ws_close_err}")


        logger.info(f"WebSocket connection cleanup complete for {websocket.client.host}:{websocket.client.port}")


# --- Main Execution ---
if __name__ == "__main__":
    import uvicorn
    # from websockets.exceptions import ConnectionClosedOK # Import needed for state check if running directly
    # from starlette.websockets import WebSocketState # Import needed for state check

    # Check if ffmpeg is available before starting
    try:
        subprocess.run(["ffmpeg", "-version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        logger.info("ffmpeg installation found.")
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.error("FATAL: ffmpeg command not found. Please ensure ffmpeg is installed and in your system's PATH.")
        exit(1)


    logger.info("Starting FastAPI server...")
    # Run with uvicorn: uvicorn main:app --host 0.0.0.0 --port 8000 --reload
    # Replace 'main' with your actual Python file name if different
    uvicorn.run(app, host="0.0.0.0", port=8000)
