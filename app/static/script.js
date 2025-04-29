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

const TARGET_SAMPLE_RATE = 16000; // Your gRPC server expects this (from SAMPLING_RATE = 16000)
const BUFFER_SIZE = 16384; // A common buffer size for ScriptProcessorNode

function logMessage(data) {
    const entry = document.createElement('div');
    entry.classList.add('log-entry');

    let content = '';
    if (data.type === 'transcription') {
        const confidenceScore = data.confidence ? data.confidence.toFixed(2) : 'N/A';
        content = `<span class="${data.is_final ? 'transcript' : 'transcript interim'}">Transcript: ${data.transcript} </span>`;
        if (data.translation) {
            content += `<br><span class="translation">Translation: ${data.translation}</span>`;
        }

    } else if (data.type === 'error') {
        content = `<span class="error">Error: ${data.message}</span>`;
    } else if (data.type === 'info') {
         content = `<span>INFO: ${data.message}</span>`;
    } else {
        content = `<span>${JSON.stringify(data)}</span>`;
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
    logMessage({ type: 'info', message: 'Attempting to play synthesized audio...' });
    audioPlayer.src = `data:audio/mp3;base64,${base64String}`;
    audioPlayer.play()
        .then(() => logMessage({ type: 'info', message: 'Audio playback started.' }))
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
            sampleRate: TARGET_SAMPLE_RATE // Try to request the target sample rate
        });
        
        // Check if the context actually got the desired sample rate
        if (audioContext.sampleRate !== TARGET_SAMPLE_RATE) {
            console.warn(`AudioContext running at ${audioContext.sampleRate}Hz, not desired ${TARGET_SAMPLE_RATE}Hz. Resampling will occur if source is different or this ScriptProcessorNode will output at this rate.`);
            // The ScriptProcessorNode will output at audioContext.sampleRate
        }


        const source = audioContext.createMediaStreamSource(mediaStream);
        scriptProcessorNode = audioContext.createScriptProcessor(BUFFER_SIZE, 1, 1); // Buffer size, input channels, output channels

        scriptProcessorNode.onaudioprocess = (audioProcessingEvent) => {
            if (!websocket || websocket.readyState !== WebSocket.OPEN) return;

            const inputBuffer = audioProcessingEvent.inputBuffer;
            const pcmData = inputBuffer.getChannelData(0); // Float32 PCM data

            // Your original client sent paFloat32. A Float32Array's underlying buffer can be sent directly.
            // The server side (gRPC client manager) will receive these bytes and should forward them.
            // The gRPC server's TranscriptionServer must be able to handle these Float32 PCM bytes.
            // If it expects 16-bit PCM, conversion would be needed here:
            // const int16Pcm = new Int16Array(pcmData.length);
            // for (let i = 0; i < pcmData.length; i++) {
            //     let s = Math.max(-1, Math.min(1, pcmData[i]));
            //     int16Pcm[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
            // }
            // websocket.send(int16Pcm.buffer);
            
            websocket.send(pcmData.buffer); // Send ArrayBuffer of Float32
        };

        source.connect(scriptProcessorNode);
        scriptProcessorNode.connect(audioContext.destination); // Necessary for onaudioprocess to fire

        const selectedLanguage = languageSelect.value;
        const wsProtocol = window.location.protocol === "https:" ? "wss:" : "ws:";
        const wsUrl = `${wsProtocol}//${window.location.host}/ws/stt?language=${selectedLanguage}`;
        websocket = new WebSocket(wsUrl);

        websocket.onopen = () => {
            updateStatus(`Connected to server. Recording started... Language: ${selectedLanguage}`);
            logMessage({type: 'info', message: `WebSocket open. Language: ${selectedLanguage}`});
        };

        websocket.onmessage = (event) => {
            try {
                const data = JSON.parse(event.data);
                logMessage(data);
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
            logMessage({type: 'error', message: `WebSocket error: ${error.message || 'Unknown error'}`});
            console.error("WebSocket error:", error);
            cleanupAudioResources(); // Also cleanup on error
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
        scriptProcessorNode.disconnect();
        scriptProcessorNode = null;
    }
    if (mediaStream) {
        mediaStream.getTracks().forEach(track => track.stop());
        mediaStream = null;
    }
    if (audioContext && audioContext.state !== 'closed') {
        audioContext.close().catch(e => console.error("Error closing AudioContext:", e));
        audioContext = null;
    }
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