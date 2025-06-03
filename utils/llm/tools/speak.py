import asyncio

import sounddevice as sd

from utils.speech_synthesis import KokoroSynthesis as Synthesis

synth = Synthesis()
SAMPLE_RATE = 24000

func_doc =\
{
    'type': 'function',
    'function':\
    {
        'name': 'speak',
        'description': '通过TTS模型合成并播放语音，用于生成语音回复',
        'parameters':\
        {
            'type': 'object',
            'properties':\
            {
                'text':\
                {
                    'type': 'str',
                    'description': '需要合成的文本，主体为中文，可包含英文'
                },
            },
            'required': ['text']
        },
    }
}

def speak(text: str):
    try:
        audio_data = asyncio.run(synth.synthesize(text))
        sd.play(audio_data, SAMPLE_RATE)
        sd.wait()
        return f"合成并播放语音成功: '{text}'"
    except Exception as e:
        return f"合成并播放语音失败: {str(e)}"
    