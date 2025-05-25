import os
import sys
import time
import argparse
import logging

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from Modality.core.modality_manager import ModalityManager
from Modality.speech.speech_recognition import SpeechRecognition
from Modality.core.error_codes import SUCCESS, get_error_message

logging.basicConfig(
    level=logging.DEBUG if os.environ.get('MODALITY_DEBUG', '0') == '1' else logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='demo_speech_recognition.log',
    filemode='w'
)
logger = logging.getLogger('DemoSpeechRecognition')

def parse_args():
    parser = argparse.ArgumentParser(description='智能座舱语音识别演示')
    parser.add_argument('--no-wake', action='store_true', help='关闭唤醒词功能')
    parser.add_argument('--no-sv', action='store_true', help='关闭声纹识别功能')
    parser.add_argument('--add-wake', type=str, help='添加自定义唤醒词')
    parser.add_argument('--register', action='store_true', help='强制进入声纹注册模式')
    parser.add_argument('--register-name', type=str, help='注册声纹的用户名')
    parser.add_argument('--max-temp', type=int, help='设置最大临时声纹数量', default=10)
    parser.add_argument('--debug', action='store_true', help='开启调试模式')
    return parser.parse_args()

def main():
    args = parse_args()
    
    if args.debug:
        os.environ["MODALITY_DEBUG"] = "1"
        logger.info("调试模式已开启")
    else:
        os.environ["MODALITY_DEBUG"] = "0"
    
    manager = ModalityManager()
    
    print("正在初始化智能座舱语音识别系统...")
    speech_modality = SpeechRecognition(name="speech_recognition")
    print("语音识别模态创建成功")
    
    result = manager.register_modality(speech_modality)
    if result != SUCCESS:
        logger.error(f"注册语音识别模态失败: {get_error_message(result)}")
        return
    
    print("语音识别模态注册成功")
    
    result = manager.start_modality("speech_recognition")
    if result != SUCCESS:
        logger.error(f"启动语音识别模态失败: {get_error_message(result)}")
        return
    
    print("语音识别模态启动成功")

    if args.no_wake:
        speech_modality.toggle_wake_word(False)
    
    if args.add_wake:
        speech_modality.add_wake_word(args.add_wake)
    
    speech_modality.set_max_temp_speakers(args.max_temp)

    if args.register:
        name = args.register_name if args.register_name else "新用户"
        speech_modality.register_speaker(name)
    
    print("系统初始化完成")

    try:
        print("\n" + "-" * 40)
        print("智能座舱语音识别系统已启动")
        print("按 Ctrl+C 停止系统")
        print("-" * 40)
        
        print("\n可用命令:")
        print("  r - 注册新的声纹")
        print("  d - 删除声纹")
        print("  w - 切换唤醒词功能")
        print("  a - 添加新的唤醒词")
        print("  k - 添加新的关键词")
        print("  t - 查看临时声纹列表")
        print("  s - 查看注册声纹列表")
        print("  q - 退出系统")
        
        last_recognized_text = ""
        
        while True:
            state = speech_modality.update()
            if state and state.recognition["text"]:
                current_text = state.recognition["text"]
                
                if current_text != last_recognized_text:
                    last_recognized_text = current_text
                    
                    result_text = []
                    if state.recognition["has_wake_word"]:
                        result_text.append(f"[唤醒词: {state.recognition['wake_word']}]")
                    if state.recognition["has_keyword"]:
                        result_text.append(f"[关键词: {state.recognition['keyword']} 类别: {state.recognition['keyword_category']}]")
                    if state.recognition["speaker_id"]:
                        result_text.append(f"[说话人: {state.recognition['speaker_name']}]")
                        
                    result_header = " ".join(result_text)
                    if result_header:
                        print(f"\n{result_header}")
                    print(f"识别结果: {current_text}")
            
            import msvcrt
            if msvcrt.kbhit():
                cmd = msvcrt.getch().decode('utf-8', errors='ignore').lower()
                print(f"\n输入命令: {cmd}")
                
                if cmd == 'r':
                    name = input("请输入用户名: ")
                    speech_modality.register_speaker(name)
                elif cmd == 'd':
                    speakers = speech_modality.get_registered_speakers()
                    if not speakers:
                        print("没有已注册的声纹")
                        continue
                        
                    print("已注册声纹:")
                    for idx, (speaker_id, info) in enumerate(speakers.items()):
                        print(f"  {idx+1}. {info['name']} (ID: {speaker_id})")
                        
                    choice = input("请输入要删除的声纹编号: ")
                    try:
                        idx = int(choice) - 1
                        if 0 <= idx < len(speakers):
                            speaker_id = list(speakers.keys())[idx]
                            if speech_modality.delete_speaker(speaker_id):
                                print(f"已删除声纹 {speakers[speaker_id]['name']}")
                        else:
                            print("无效的编号")
                    except ValueError:
                        print("请输入有效的数字")
                elif cmd == 'w':
                    enabled = speech_modality.toggle_wake_word()
                    print(f"唤醒词功能已{'启用' if enabled else '关闭'}")
                elif cmd == 'a':
                    wake_word = input("请输入要添加的唤醒词: ")
                    if speech_modality.add_wake_word(wake_word):
                        print(f"已添加唤醒词: {wake_word}")
                elif cmd == 'k':
                    keyword = input("请输入要添加的关键词: ")
                    category = input("请输入关键词类别: ")
                    if speech_modality.add_keyword(keyword, category):
                        print(f"已添加关键词: {keyword} -> {category}")
                elif cmd == 't':
                    temp_speakers = speech_modality.get_temp_speakers()
                    if not temp_speakers:
                        print("没有临时声纹")
                    else:
                        print("临时声纹列表:")
                        for temp_id, info in temp_speakers.items():
                            print(f"  {info['name']} (ID: {temp_id}, 创建时间: {info.get('created', 'N/A')})")
                            
                        promote = input("是否提升某个临时声纹为正式声纹? (y/n): ").lower()
                        if promote == 'y':
                            temp_id = input("请输入临时声纹ID: ")
                            if temp_id in temp_speakers:
                                name = input("请输入新的用户名 (直接回车使用原名): ")
                                name = name if name else None
                                new_id = speech_modality.promote_temp_speaker(temp_id, name)
                                if new_id:
                                    print(f"临时声纹已提升为正式声纹，新ID: {new_id}")
                            else:
                                print(f"临时声纹 {temp_id} 不存在")
                elif cmd == 's':
                    speakers = speech_modality.get_registered_speakers()
                    if not speakers:
                        print("没有已注册的声纹")
                    else:
                        print("已注册声纹列表:")
                        for speaker_id, info in speakers.items():
                            print(f"  {info['name']} (ID: {speaker_id})")
                elif cmd == 'q':
                    break

            time.sleep(0.1)
    
    except KeyboardInterrupt:
        print("\n检测到终止信号")
    finally:
        print("正在关闭系统...")
        manager.shutdown_all()
        print("系统已关闭")

if __name__ == "__main__":
    main()
    