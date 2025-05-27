#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# _author = 'Xianda Tang' 'Yidian Lin'

"""
Module Description:
    语音播放器
"""

import asyncio
import sounddevice as sd
from .speech_synthesis.kokoro_synthesis import KokoroSynthesis as SpeechSyn

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

    async def speech_synthesize(self, text: str):
        """异步合成语音,返回音频流(numpy数组)"""
        audio_data = await self.synth.synthesize(text)
        return audio_data

speecher_player = SpeecherPlayer()
