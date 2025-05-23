import time
from Modality import ModalityManager
from Modality import HeadPoseTrackerGRU as HeadPoseTracker
from Modality import GestureTracker
from Modality import SpeechRecognition

from Modality import HeadPoseState, GestureState, SpeechState

manager = ModalityManager()
#head_tracker = HeadPoseTracker()
#gesture_tracker = GestureTracker()
speech_tracker = SpeechRecognition()

#manager.register_modality(head_tracker)
#manager.register_modality(gesture_tracker)
manager.register_modality(speech_tracker)

modalities: dict = manager.get_all_modalities()

for name in modalities.keys():
    print(name)
    manager.start_modality(name)

last_recognized_text = ""
        
while True:
    state = manager.update_all()
    if state and state.recognition["text"]:
        current_text = state.recognition["text"]
                
    if current_text != last_recognized_text:
        last_recognized_text = current_text
                    
        result_text = []
        #            if state.recognition["has_wake_word"]:
        #                result_text.append(f"[唤醒词: {state.recognition['wake_word']}]")
        #            if state.recognition["has_keyword"]:
        #                result_text.append(f"[关键词: {state.recognition['keyword']} 类别: {state.recognition['keyword_category']}]")
        #            if state.recognition["speaker_id"]:
        #                result_text.append(f"[说话人: {state.recognition['speaker_name']}]")
                        
        result_header = " ".join(result_text)
        if result_header:
            print(f"\n{result_header}")
            print(f"识别结果: {current_text}")