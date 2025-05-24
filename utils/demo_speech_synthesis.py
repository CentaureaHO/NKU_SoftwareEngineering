import os
import asyncio
import sounddevice as sd
import numpy as np
import time
from tools import speech_synthesize, speech_synthesize_sync

SAMPLE_RATE = 24000

async def test_async():
    text = "这是一个异步合成的语音示例。"
    print(f"合成文本: '{text}'")
    
    start_time = time.time()
    audio_data = await speech_synthesize(text)
    end_time = time.time()
    
    print(f"合成完成，时长: {end_time - start_time:.2f}秒")
    print(f"音频长度: {len(audio_data)/SAMPLE_RATE:.2f}秒")
    
    print("播放音频...")
    sd.play(audio_data, SAMPLE_RATE)
    sd.wait()
    print("播放结束")

def test_sync():
    text = "这是一个同步合成的语音示例。"
    print(f"合成文本: '{text}'")
    
    start_time = time.time()
    audio_data = speech_synthesize_sync(text)
    end_time = time.time()
    
    print(f"合成完成，时长: {end_time - start_time:.2f}秒")
    print(f"音频长度: {len(audio_data)/SAMPLE_RATE:.2f}秒")
    
    print("播放音频...")
    sd.play(audio_data, SAMPLE_RATE)
    sd.wait()
    print("播放结束")

async def async_main():
    print("开始异步合成测试...")
    await test_async()
    print("异步合成测试完成")

def sync_main():
    print("开始同步合成测试...")
    test_sync()
    print("同步合成测试完成")

if __name__ == "__main__":
    # asyncio.run(async_main())
    sync_main()
