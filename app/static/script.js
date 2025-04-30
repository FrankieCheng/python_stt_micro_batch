// app/static/script.js
const startRecBtn = document.getElementById('startRecBtn');
const stopRecBtn = document.getElementById('stopRecBtn');
const languageSelect = document.getElementById('languageSelect');
const statusDiv = document.getElementById('status');
const outputLog = document.getElementById('outputLog');

let websocket;
let audioContext;
let mediaStream;
let scriptProcessorNode;
let audioPlayer = new Audio(); // For playing received MP3s

const TARGET_SAMPLE_RATE = 16000;
// const BUFFER_SIZE = 4096; // For ~256ms chunks from client
// const BUFFER_SIZE = 8192;   // For ~512ms chunks from client (as per recent discussion)
const BUFFER_SIZE = 32768; // Current in your file: for ~1024ms chunks from client

let lastSendTime = 0; // To track send time for RTT (approximate)

function logMessage(data) {
    const entry = document.createElement('div');
    entry.classList.add('log-entry');

    let content = '';
    if (data.type === 'transcription') {
        const confidenceScore = data.confidence ? data.confidence.toFixed(2) : 'N/A';
        
        // --- Timing Information Display ---
        let timingDetails = [];
        if (data.stt_duration_ms) {
            timingDetails.push(`STT: ${data.stt_duration_ms}ms`);
        }
        if (data.translation_duration_ms) { // Assuming server might send this
            timingDetails.push(`Translate: ${data.translation_duration_ms}ms`);
        }
        if (data.tts_duration_ms) {
            timingDetails.push(`TTS: ${data.tts_duration_ms}ms`);
        }
        // Approximate RTT if we had sendTime (more complex to correlate accurately in stream)
        // if (data.is_final && lastSendTime > 0) {
        //     const rtt = performance.now() - lastSendTime;
        //     timingDetails.push(`Approx. RTT: ${rtt.toFixed(0)}ms`);
        //     lastSendTime = 0; // Reset for next final segment
        // }
        
        let timingString = timingDetails.length > 0 ? ` <small>[${timingDetails.join(', ')}]</small>` : '';
        // --- End Timing Information Display ---

        content = `<span class="${data.is_final ? 'transcript' : 'transcript interim'}">Transcript: ${data.transcript}${timingString}</span>`;
        if (data.translation) {
            content += `<br><span class="translation">Translation: ${data.translation}</span>`;
        }

    } else if (data.type === 'error') {
        content = `<span class="error">Error: ${data.message}</span>`;
    } else if (data.type === 'info') {
        entry.classList.add('info'); // Add this line
        content = `<span>INFO: ${data.message}</span>`;
    } else {
        // For debugging, show the whole data object if it's an unknown type
        content = `<span>UNKNOWN DATA: ${JSON.stringify(data)}</span>`;
        console.warn("Received unknown data structure:", data);
    }
    entry.innerHTML = content;
    outputLog.appendChild(entry);
    outputLog.scrollTop = outputLog.scrollHeight;

    if (data.type === 'transcription' && data.is_final && data.audio_data_b64 && data.audio_format === 'mp3') {
        playMp3FromBase64(data.audio_data_b64);
    }
}

function updateStatus(message) {
    statusDiv.textContent = `Status: ${message}`;
}

function playMp3FromBase64(base64String) {
    const playbackStartTime = performance.now();
    logMessage({ type: 'info', message: 'Attempting to play synthesized audio...' });
    audioPlayer.src = `data:audio/mp3;base64,${base64String}`;
    audioPlayer.play()
        .then(() => {
            const playbackSetupTime = performance.now() - playbackStartTime;
            logMessage({ type: 'info', message: `Audio playback started. (Setup: ${playbackSetupTime.toFixed(0)}ms)` });
        })
        .catch(e => {
            logMessage({ type: 'error', message: `Audio playback error: ${e.message}` });
            console.error("Audio playback error:", e);
        });
}


startRecBtn.onclick = async () => {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        logMessage({type: 'error', message: 'getUserMedia API not supported in this browser.'});
        return;
    }

    startRecBtn.disabled = true;
    stopRecBtn.disabled = false;
    outputLog.innerHTML = ''; // Clear previous logs
    updateStatus('Requesting microphone access...');

    try {
        mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
        updateStatus('Microphone access granted. Initializing audio processing...');

        audioContext = new (window.AudioContext || window.webkitAudioContext)({
            sampleRate: TARGET_SAMPLE_RATE
        });
        
        if (audioContext.sampleRate !== TARGET_SAMPLE_RATE) {
            console.warn(`AudioContext running at ${audioContext.sampleRate}Hz, not target ${TARGET_SAMPLE_RATE}Hz. Input will be at ${audioContext.sampleRate}Hz from ScriptProcessorNode.`);
        }

        const source = audioContext.createMediaStreamSource(mediaStream);
        // The bufferSize for createScriptProcessor must be a power of 2, from 256 to 16384.
        // 4096 samples @ 16kHz = 256ms
        // 8192 samples @ 16kHz = 512ms
        // 16384 samples @ 16kHz = 1024ms
        scriptProcessorNode = audioContext.createScriptProcessor(BUFFER_SIZE, 1, 1);

        scriptProcessorNode.onaudioprocess = (audioProcessingEvent) => {
            if (!websocket || websocket.readyState !== WebSocket.OPEN) return;

            const inputBuffer = audioProcessingEvent.inputBuffer;
            // Get data for the first channel
            const pcmData = inputBuffer.getChannelData(0); // Float32 PCM data

            // If input sample rate is different from TARGET_SAMPLE_RATE, resampling is needed here or on server.
            // For simplicity, assuming inputBuffer.sampleRate matches TARGET_SAMPLE_RATE due to context hint.
            // If not, pcmData is at audioContext.sampleRate.

            // lastSendTime = performance.now(); // For RTT - more complex to correlate
            websocket.send(pcmData.buffer); 
        };

        source.connect(scriptProcessorNode);
        scriptProcessorNode.connect(audioContext.destination); // Necessary for onaudioprocess to fire

        const selectedLanguage = languageSelect.value;
        const wsProtocol = window.location.protocol === "https:" ? "wss:" : "ws:";
        const wsUrl = `${wsProtocol}//${window.location.host}/ws/stt?language=${selectedLanguage}`;
        websocket = new WebSocket(wsUrl);

        websocket.onopen = () => {
            updateStatus(`Connected. Recording started... Language: ${selectedLanguage}. Sending audio in ~${((BUFFER_SIZE / TARGET_SAMPLE_RATE) * 1000).toFixed(0)}ms chunks.`);
            logMessage({type: 'info', message: `WebSocket open. Language: ${selectedLanguage}`});
        };

        websocket.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                logMessage(data); // logMessage will now display timings if present
            } catch (e) {
                logMessage({type: 'error', message: 'Received malformed JSON from server.'});
                console.error("Error parsing server message:", e, event.data);
            }
        };

        websocket.onclose = (event) => {
            updateStatus(`Disconnected. ${event.reason || ''}`);
            logMessage({type: 'info', message: `WebSocket closed. Code: ${event.code}, Reason: ${event.reason}`});
            cleanupAudioResources();
        };

        websocket.onerror = (error) => {
            updateStatus('WebSocket error. See console.');
            const errorMessage = error.message || (error.target && error.target.url ? `Could not connect to ${error.target.url}` : 'Unknown WebSocket error');
            logMessage({type: 'error', message: `WebSocket error: ${errorMessage}`});
            console.error("WebSocket Error: ", error);
            cleanupAudioResources();
        };

    } catch (err) {
        updateStatus(`Error: ${err.message}`);
        logMessage({type: 'error', message: `Failed to start recording: ${err.name} - ${err.message}`});
        console.error("Error starting recording:", err);
        cleanupAudioResources();
    }
};

function cleanupAudioResources() {
    if (scriptProcessorNode) {
        scriptProcessorNode.onaudioprocess = null; // Stop processing audio
        scriptProcessorNode.disconnect();
        scriptProcessorNode = null;
    }
    if (mediaStream) {
        mediaStream.getTracks().forEach(track => track.stop());
        mediaStream = null;
    }
    // It's generally better not to close/recreate audioContext too often unless necessary.
    // If you do, ensure it's fully stopped/closed.
    // if (audioContext && audioContext.state !== 'closed') {
    //     audioContext.close().catch(e => console.error("Error closing AudioContext:", e));
    //     audioContext = null;
    // }

    if (websocket && (websocket.readyState === WebSocket.OPEN || websocket.readyState === WebSocket.CONNECTING)) {
        websocket.close(1000, "Client cleanup");
    }
    websocket = null;

    startRecBtn.disabled = false;
    stopRecBtn.disabled = true;
}

stopRecBtn.onclick = () => {
    updateStatus('Stopping recording...');
    cleanupAudioResources();
    logMessage({type: 'info', message: 'Recording stopped by user.'});
};

audioPlayer.onended = () => {
    logMessage({ type: 'info', message: 'Audio playback finished.' });
};
audioPlayer.onerror = (e) => {
    logMessage({ type: 'error', message: `Audio playback failed: ${audioPlayer.error?.message || 'Unknown audio error'}` });
    console.error("Audio player error", audioPlayer.error);
};

updateStatus('Idle. Select language and press Start Recording.');