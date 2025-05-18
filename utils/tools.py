import asyncio
from speech_synthesis.kokoro_synthesis import KokoroSynthesis as SpeechSyn

synth = SpeechSyn()

async def speech_synthesize(text: str):
    """异步合成语音，返回音频流（numpy数组）"""
    audio_data = await synth.synthesize(text)
    return audio_data

def speech_synthesize_sync(text: str):
    """同步合成语音，返回音频流"""
    return asyncio.run(synth.synthesize(text))
