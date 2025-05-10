import asyncio
from .speech_synthesis.kokoro_synthesis import KokoroSynthesis as SpeechSyn
import sounddevice as sd

# synth = SpeechSyn()

class SpeecherPlayer:
    """语音播放器类:合成并播放语音"""
    def __init__(self):
        self.synth = SpeechSyn()

    def speech_synthesize_sync(self,text: str):
        """同步合成语音，返回音频流"""
        print(f"合成文本: '{text}'")
        audio_data = asyncio.run(self.synth.synthesize(text))
        SAMPLE_RATE = 24000
        sd.play(audio_data, SAMPLE_RATE)
        sd.wait()
        #return asyncio.run(synth.synthesize(text))

    # TODO():后续可能需要异步播放语音
    # async def speech_synthesize(text: str):
    #     """异步合成语音，返回音频流（numpy数组）"""
    #     audio_data = await synth.synthesize(text)
    #     return audio_data

speecher_player = SpeecherPlayer()

if __name__ == "__main__":
    speecher_player.speech_synthesize_sync("这是一个同步合成的语音示例。")
