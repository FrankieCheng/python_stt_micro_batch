import gradio as gr
from transcribe_server import TranscriptionServer
import argparse
import uuid
import pylru
import logging

FORMAT = '%(levelname)s: %(asctime)s: %(message)s'
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('GradioTest')

CACHE_SIZE = 100
transcription_server_cache = pylru.lrucache(CACHE_SIZE)
transcription_result_cache = pylru.lrucache(CACHE_SIZE)

def transcribe_chunks(uuid_value, new_chunk, source_language="Chinese (Simplified, China)"):
    transcript_stream_response = transcription_server_cache[uuid_value].recv_audio(new_chunk=new_chunk, language_code=language_mappings[source_language]);
    history_transcript_results = transcription_result_cache[uuid_value] if uuid_value in transcription_result_cache else []
    if len(history_transcript_results) > 0 and not history_transcript_results[len(history_transcript_results)-1].is_final:
        history_transcript_results = history_transcript_results[0:len(history_transcript_results)-1]
    
    if None != transcript_stream_response and transcript_stream_response.results and transcript_stream_response.results[0].alternatives:
        speech_event_offset = transcript_stream_response.speech_event_offset
        results = transcript_stream_response.results
        for result in results:
            history_transcript_results.append(result)
    transcription_result_cache[uuid_value] = history_transcript_results
    current_transcript = ""
    for result in history_transcript_results:
        if result.alternatives and result.alternatives[0].transcript:
            current_transcript = current_transcript + result.alternatives[0].transcript + ("\n" if result.is_final else "")

    return uuid_value, current_transcript

def generate_uuid():
    uuid_value = str(uuid.uuid4())
    print(f"generate uuid = {uuid_value}")
    return uuid_value

language_codes = [
    "en-US",  # English (United States)
    "zh-Hans-CN",  # Chinese (Simplified, China)
    "ja-JP",  # Japanese (Japan)
    "de-DE",  # German (Germany)
    "fr-FR",  # French (France)
    "es-ES",  # Spanish (Spain)
    "pt-BR",  # Portuguese (Brazil)
    "ru-RU",  # Russian (Russia)
    "hi-IN",  # Hindi (India)
    "ar-EG",  # Arabic (Egypt)
]

language_names = ['English (United States)', 'Chinese (Simplified, China)', 'Japanese (Japan)', 'German (Germany)', 'French (France)', 'Spanish (Spain)', 'Portuguese (Brazil)', 'Russian (Russia)', 'Hindi (India)', 'Arabic (Egypt)']

language_mappings = {
    "English (United States)": "en-US",
    "Chinese (Simplified, China)": "zh-Hans-CN",
    "Japanese (Japan)": "ja-JP",
    "German (Germany)": "de-DE",
    "French (France)": "fr-FR",
    "Spanish (Spain)": "es-ES",
    "Portuguese (Brazil)": "pt-BR",
    "Russian (Russia)": "ru-RU",
    "Hindi (India)": "hi-IN",
    "Arabic (Egypt)": "ar-EG",
}

def start_recording(uuid_value, new_chunk, source_language="Chinese (Simplified, China)"):
    print(f"start recording. init uuid_value.")
    uuid_value = generate_uuid()
    transcription_server_cache[uuid_value] = TranscriptionServer(project_id=PROJECT_ID, location=LOCATION, recognizer="-")
    return uuid_value, ""
    #global transcription_server
    #transcription_server = TranscriptionServer(project_id=PROJECT_ID, location=LOCATION, recognizer="-")

def stop_recording(uuid_value, new_chunk, source_language="Chinese (Simplified, China)"):
    print("stop recording.")
    #global transcription_server
'''
:return: Gradio interface instance
'''

'''
:return: Gradio interface instance
'''

parser = argparse.ArgumentParser(description='SpeechToText service')
parser.add_argument('-project', action='store', dest='project', type=str, default='',
    help='project')
parser.add_argument('-location', action='store', dest='location', type=str, default='us-central1',
    help='location')
args = parser.parse_args()

if args.project == '':
    raise Exception ('Please use "-project " input your project id.')

PROJECT_ID = args.project
LOCATION = args.location
#transcription_server = TranscriptionServer(project_id=args.project, location= args.location, recognizer="-")

with gr.Blocks() as demo:
    grMicrophoneAudio = gr.Audio(sources=["microphone"], label="Please Speak: ", type="filepath", streaming=True)
    language_dropdown = gr.Dropdown(choices=language_names, label="Source Language", value="Chinese (Simplified, China)")
    transcript_textbox = gr.Textbox(lines=10, label="Transcript")
    # stream = gr.State()
    #out = gr.Audio()
    #uuid_textbox = gr.Textbox(lines=10, label="uuid", visible=False, value="")
    stats = gr.State(value="")
    app = gr.Interface(
        fn = transcribe_chunks, 
        title="Micro-Batch Transcription For STT with Gemini/Chirp_2",
        description="Please select the source language first, then speak and click the \"Transcribe\"button.",
        inputs = [
            stats, 
            grMicrophoneAudio,
            language_dropdown
            ],
        outputs=[stats, transcript_textbox],
        live = True,
    )

    grMicrophoneAudio.start_recording(start_recording, inputs=[stats, grMicrophoneAudio,
            language_dropdown], outputs=[stats, transcript_textbox],)
demo.launch(server_port=7860,share=True)
