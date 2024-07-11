#  Speech-to-Text GRPC to mock the stream requests/responses for users to transcript microphone/file stream.
Bi-directional streaming speech-to-text (STT or ASR) to mock the stream requests from microphone/file stream, and response the transcript contents to end users.

It is recommended to use Linux to deploy the server can avoid lots of compatibility issues.

# Clone the repo
Clone the repo from Git.

# Install Python library dependicies
Preferred way is to do in virtualenv (Python 3.10+).
```
pip3 install -r requirements.txt
```

# Compile .proto to generate grpc files.
Skip if you only don't change the proto, since stt_pb2.py/stt_pb2_grpc.py are already generated.
```
cd proto
bash generate_pb.sh
```

## Start server
Start the server on a given port/gcp project_id/location. 
```
python3 stt_server.py -p {port} -project {PROJECT_ID} -location {location}
```

## Stream from recorded file
```
python3 test_client_file_stream.py -p {port} -project {PROJECT_ID} -location {location} -l zh-Hans-CN -in temp-test-chinese.wav
```

## Stream from microphone
```
python3 test_client_microphone.py -p 9080 -l zh-Hans-CN -project {PROJECT_ID} -location {location}
```

## Gradio Demo
```
python3 gradio-test.py -l zh-Hans-CN -project {PROJECT_ID} -location {location}
```

## Development
Users need to access to specific services, you can refer to the code:test_client_microphone.py.
The server use 'Gemini' as transcript engine, and you can switch the engine to 'Chirp_2', change 'transcribe_by_gemini' to 'transcribe' in ['transcribe_server.py'](transcribe_server.py) line 203.