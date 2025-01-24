
from google.cloud import texttospeech_v1 as texttospeech
import os
import pyaudio
from loguru import logger


os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./cloudplus1-test-new-f7f503ac2fd0.json"
# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/home/ubuntu/python_stt_micro_batch-main/cloudplus1-test-new-f7f503ac2fd0.json"
# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/home/ubuntu/python/python/cloudplus1-test01-a19aa208ef3a.json"


audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MULAW,
        sample_rate_hertz=8000,
        speaking_rate=1.0,  # 正常语速
        pitch=0.0,  # 正常音高
    )

# 配置语音
voice_config = texttospeech.VoiceSelectionParams(
    language_code="zh-cmn",
    ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL,  # 中性语音
)


class TextToSpeechBridge:
    def __init__(self,on_audio_response):
        """
        初始化文本转音频桥接器。

        :param audio_config: 音频配置，用于设置音频格式等。
        :param voice_config: 语音配置，用于选择语音特性。
        :param on_audio_response: 音频生成后的回调函数，处理音频流。
        """
        self._on_audio_response = on_audio_response
        self.audio_config = audio_config
        self.voice_config = voice_config

    def start(self,text):
        """
        开始处理文本并生成音频流。
        """
        try:
            client = texttospeech.TextToSpeechClient()
            # 请求 Google Text-to-Speech 服务
            logger.success(text)
            input_text = texttospeech.SynthesisInput(text=text)
            response = client.synthesize_speech(
                input=input_text,
                voice=self.voice_config,
                audio_config=self.audio_config,
            )
            # logger.debug(response.audio_content[58:])
            # self._on_audio_response(response.audio_content[58:],text)
            return response.audio_content[58:]

        except Exception as e:
            logger.error(e)
            return ''


# 示例回调函数，用于处理生成的音频流
def audio_response_handler(audio_content,text):
    """
    处理生成的音频流（如保存到文件或播放）。

    :param audio_content: Google TTS 返回的音频内容。
    """

    # 配置音频参数
    FORMAT = pyaudio.paInt16  # 音频数据格式（16-bit PCM，常见格式）
    CHANNELS = 1  # 声道数量（1 为单声道，2 为立体声）
    RATE = 8000  # 采样率（单位：Hz）

    # 初始化 PyAudio
    audio = pyaudio.PyAudio()

    # 打开输出流
    stream = audio.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        output=True)

    # 播放音频流
    chunk_size = 1024  # 每次写入的音频块大小
    start = 0
    while start < len(audio_content):
        end = start + chunk_size
        stream.write(audio_content[start:end])  # 写入音频数据到输出流
        start = end

    # 关闭流和终端
    stream.stop_stream()
    stream.close()
    audio.terminate()


# 示例用法
if __name__ == "__main__":

    bridge = TextToSpeechBridge(audio_response_handler)
    # 模拟文本流输入
    try:

        for text in ["实现中华民族伟大复兴需要中华文化繁荣兴盛。", "中华文化是中华民族生存和发展的重要力量,",
                     "为中华民族克服困难、生生不息提供了强大精神支撑。","没有中华文化繁荣兴盛,就没有中华民族伟大复兴。",
                     "一个民族的复兴需要强大的物质力量,也需要强大的精神力量。",
                     "文艺是时代前进的号角,最能代表一个时代的风貌,最能引领一个时代的风气,对于实现中华民族伟大复兴具有重要作用。"]:
            bridge.start(text)
    except Exception as e:
        logger.debug(str(e))
