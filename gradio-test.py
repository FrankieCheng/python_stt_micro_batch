import gradio as gr
from transcribe_server import TranscriptionServer
import argparse

def main(stream, new_chunk, source_language="Chinese (Simplified, China)"):
    return stream, transcription_server.recv_audio(new_chunk=new_chunk, language_code=language_mappings[source_language])

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

def start_recording(audio_filepath):
    print(f"start recording.{audio_filepath}")
    global transcription_server
    global PROJECT_ID
    global LOCATION
    transcription_server = TranscriptionServer(project_id=PROJECT_ID, location=LOCATION, recognizer="-")

def stop_recording(self, audio_filepath):
    print("stop recording.")
    global transcription_server
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
transcription_server = TranscriptionServer(project_id=args.project, location= args.location, recognizer="-")

with gr.Blocks() as demo:
    grMicrophoneAudio = gr.Audio(sources=["microphone"], label="Please Speak: ", type="filepath", streaming=True)
    # stream = gr.State()
    #out = gr.Audio()
    app = gr.Interface(
        title="Micro-Batch Transcription For STT with Gemini/Chirp_2",
        description="Please select the source language first, then speak and click the \"Transcribe\"button.",
        fn = main,
        inputs = [
            "state", 
            grMicrophoneAudio,
            gr.Dropdown(choices=language_names, label="Source Language", value="Chinese (Simplified, China)")
            ],
        outputs=["state", gr.Textbox(lines=10, label="Transcript")],
        # submit_btn = gr.Button("Transcribe", visible=True),
        # allow_flagging="never",
        live=True,
    )
    grMicrophoneAudio.start_recording(start_recording, inputs=[grMicrophoneAudio])
    grMicrophoneAudio.stop_recording(stop_recording, inputs=[grMicrophoneAudio])
demo.launch(server_port=7860,share=True)
