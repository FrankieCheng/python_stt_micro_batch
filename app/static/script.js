// app/static/script.js
const startRecBtn = document.getElementById('startRecBtn');
const stopRecBtn = document.getElementById('stopRecBtn');
const languageSelect = document.getElementById('languageSelect');
const statusDiv = document.getElementById('status');
const outputLog = document.getElementById('outputLog');
const toggleTtsBtn = document.getElementById('toggleTtsBtn'); // Get the new button
const enableTtsParam = ttsPlaybackEnabled ? 'true' : 'false';
const wsUrl = `<span class="math-inline">\{wsProtocol\}//</span>{window.location.host}/ws/stt?language=<span class="math-inline">\{selectedLanguage\}&enable\_tts\=</span>{enableTtsParam}`;
websocket = new WebSocket(wsUrl);

let websocket;
let audioContext;
let mediaStream;
let scriptProcessorNode;
let audioPlayer = new Audio(); // For playing received MP3s


// --- STATE VARIABLE FOR TTS PLAYBACK ---
let ttsPlaybackEnabled = true; // Default to enabled

const TARGET_SAMPLE_RATE = 16000;
// Choose your desired client-side buffer size:
// const BUFFER_SIZE = 4096; // For ~256ms chunks from client
const BUFFER_SIZE = 8192;   // For ~512ms chunks from client
// const BUFFER_SIZE = 16384; // For ~1024ms chunks from client (was in your last version)

// --- Initialize TTS Button State and Visuals ---
function updateTtsButtonVisuals() {
    if (ttsPlaybackEnabled) {
        toggleTtsBtn.innerHTML = '<i class="fas fa-volume-up"></i> Playback On';
        toggleTtsBtn.classList.remove('tts-disabled'); // Uses main button style
        toggleTtsBtn.style.backgroundColor = "var(--info-color)"; // Explicitly set "on" color
        toggleTtsBtn.title = "Audio Playback is ON. Click to disable.";
    } else {
        toggleTtsBtn.innerHTML = '<i class="fas fa-volume-mute"></i> Playback Off';
        toggleTtsBtn.classList.add('tts-disabled'); // Uses specific CSS for "off" state
        toggleTtsBtn.style.backgroundColor = "var(--warning-color)"; // Explicitly set "off" color
        toggleTtsBtn.title = "Audio Playback is OFF. Click to enable.";
    }
}

function logMessage(data) {
    const entry = document.createElement('div');
    entry.classList.add('log-entry');

    let content = '';
    if (data.type === 'transcription') {
        const confidenceScore = data.confidence ? data.confidence.toFixed(2) : 'N/A';
        let timingDetails = [];
        if (data.stt_duration_ms) {
            timingDetails.push(`STT: ${data.stt_duration_ms}ms`);
        }
        if (data.translation_duration_ms) {
            timingDetails.push(`Translate: ${data.translation_duration_ms}ms`);
        }
        if (data.tts_duration_ms) { // This is server-side TTS generation time
            timingDetails.push(`TTS Gen: ${data.tts_duration_ms}ms`);
        }
        let timingString = timingDetails.length > 0 ? ` <small>[${timingDetails.join(', ')}]</small>` : '';

        content = `<span class="${data.is_final ? 'transcript' : 'transcript interim'}">Transcript: ${data.transcript}${timingString}</span>`;
        if (data.translation) {
            content += `<br><span class="translation">Translation: ${data.translation}</span>`;
        }

    } else if (data.type === 'error') {
        entry.classList.add('error'); // Add class for specific styling
        content = `<span class="error-message">Error: ${data.message}</span>`; // Use a class for the message itself if needed
    } else if (data.type === 'info') {
        entry.classList.add('info'); // Add class for specific styling
        content = `<span>INFO: ${data.message}</span>`;
    } else {
        content = `<span>UNKNOWN DATA: ${JSON.stringify(data)}</span>`;
        console.warn("Received unknown data structure:", data);
    }
    entry.innerHTML = content;
    outputLog.appendChild(entry);
    outputLog.scrollTop = outputLog.scrollHeight;

    // --- MODIFIED PLAYBACK CONDITION ---
    if (ttsPlaybackEnabled && data.type === 'transcription' && data.is_final && data.audio_data_b64 && data.audio_format === 'mp3') {
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
            logMessage({ type: 'info', message: `Audio playback started. (Client Play Setup: ${playbackSetupTime.toFixed(0)}ms)` });
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
    outputLog.innerHTML = ''; 
    updateStatus('Requesting microphone access...');

    try {
        mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
        updateStatus('Microphone access granted. Initializing audio processing...');

        audioContext = new (window.AudioContext || window.webkitAudioContext)({
            sampleRate: TARGET_SAMPLE_RATE
        });
        
        if (audioContext.sampleRate !== TARGET_SAMPLE_RATE) {
            console.warn(`AudioContext running at ${audioContext.sampleRate}Hz, not target ${TARGET_SAMPLE_RATE}Hz. Input will be at ${audioContext.sampleRate}Hz from ScriptProcessorNode.`);
        } else {
            console.log(`AudioContext successfully started at TARGET_SAMPLE_RATE: ${audioContext.sampleRate}Hz.`);
        }
        window.loggedInputBufferSampleRate = false; 

        const source = audioContext.createMediaStreamSource(mediaStream);
        scriptProcessorNode = audioContext.createScriptProcessor(BUFFER_SIZE, 1, 1);

        scriptProcessorNode.onaudioprocess = (audioProcessingEvent) => {
            if (!websocket || websocket.readyState !== WebSocket.OPEN) return;
            
            if (!window.loggedInputBufferSampleRate) { 
                console.log("ScriptProcessorNode InputBuffer sample rate:", audioProcessingEvent.inputBuffer.sampleRate);
                window.loggedInputBufferSampleRate = true; 
            }
            const pcmData = audioProcessingEvent.inputBuffer.getChannelData(0);
            websocket.send(pcmData.buffer); 
        };

        source.connect(scriptProcessorNode);
        scriptProcessorNode.connect(audioContext.destination);

        const selectedLanguage = languageSelect.value;
        const wsProtocol = window.location.protocol === "https:" ? "wss:" : "ws:";
        
        // --- ADD TTS PREFERENCE TO WEBSOCKET URL ---
        // This part is for the *advanced* solution where server avoids TTS generation.
        // For the current client-side only toggle, this isn't strictly needed yet,
        // but good for future extension.
        // const enableTtsParam = ttsPlaybackEnabled ? 'true' : 'false';
        // const wsUrl = `${wsProtocol}//${window.location.host}/ws/stt?language=${selectedLanguage}&enable_tts=${enableTtsParam}`;
        const wsUrl = `${wsProtocol}//${window.location.host}/ws/stt?language=${selectedLanguage}`; // Current simpler version


        websocket = new WebSocket(wsUrl);

        websocket.onopen = () => {
            updateStatus(`Connected. Language: ${selectedLanguage}. Sending audio in ~${((BUFFER_SIZE / TARGET_SAMPLE_RATE) * 1000).toFixed(0)}ms chunks.`);
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
        scriptProcessorNode.onaudioprocess = null; 
        scriptProcessorNode.disconnect();
        scriptProcessorNode = null;
    }
    if (mediaStream) {
        mediaStream.getTracks().forEach(track => track.stop());
        mediaStream = null;
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

// --- TTS TOGGLE BUTTON LOGIC ---
if (toggleTtsBtn) { // Check if button exists
    toggleTtsBtn.onclick = () => {
        ttsPlaybackEnabled = !ttsPlaybackEnabled; // Toggle the state
        updateTtsButtonVisuals();
        logMessage({ type: 'info', message: `Audio Playback ${ttsPlaybackEnabled ? 'Enabled' : 'Disabled'}.` });
        // Optional: Store preference in localStorage
        // localStorage.setItem('ttsPlaybackEnabled', ttsPlaybackEnabled);
    };
}

// --- Initialize button text/icon on page load ---
document.addEventListener('DOMContentLoaded', (event) => {
    // Optional: Load preference from localStorage
    // const savedTtsPref = localStorage.getItem('ttsPlaybackEnabled');
    // if (savedTtsPref !== null) {
    //     ttsPlaybackEnabled = JSON.parse(savedTtsPref);
    // }
    if (toggleTtsBtn) { // Check if button exists before updating
        updateTtsButtonVisuals();
    }
    updateStatus('Idle. Select language and press Start.');
});


audioPlayer.onended = () => {
    logMessage({ type: 'info', message: 'Audio playback finished.' });
};
audioPlayer.onerror = (e) => {
    logMessage({ type: 'error', message: `Audio playback failed: ${audioPlayer.error?.message || 'Unknown audio error'}` });
    console.error("Audio player error", audioPlayer.error);
};