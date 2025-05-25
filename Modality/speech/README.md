# 智能座舱语音识别模块

智能座舱语音识别模块(`SpeechRecognition`)继承自BaseModality，提供了唤醒词识别、特定关键词识别和声纹识别等功能，适用于智能座舱的语音交互系统。

## 主要特性

- **多唤醒词支持**：可同时配置多个唤醒词（如"你好小智"、"小爱同学"等），支持添加和删除
- **特定关键词识别**：支持定义特定领域的关键词（如"打开车窗"、"关闭空调"等）及其分类
- **声纹识别与管理**：支持注册、识别、删除声纹，提供个性化交互体验
- **临时声纹存储**：对未匹配到的声纹自动保存为临时文件，支持后续提升为正式声纹
- **离线运行支持**：模型首次下载后会缓存到本地，支持断网环境下运行

## 使用方法

### 基本用法

```python
from Modality.speech.speech_recognition import SpeechRecognition
from Modality.core.modality_manager import ModalityManager

# 创建语音识别模态
speech = SpeechRecognition(name="speech_recognition")

# 注册并启动模态
manager = ModalityManager()
manager.register_modality(speech)
manager.start_modality("speech_recognition")

while True:
    state = speech.update()
    if state and state.recognition["text"]:
        print(f"识别结果: {state.recognition['text']}")
        
        if state.recognition["has_wake_word"]:
            print(f"检测到唤醒词: {state.recognition['wake_word']}")
            
        if state.recognition["has_keyword"]:
            print(f"检测到关键词: {state.recognition['keyword']}, 类别: {state.recognition['keyword_category']}")
```

### 唤醒词管理

```python
# 操作唤醒词
speech.add_wake_word("你好小车")
speech.remove_wake_word("你好小智")

# 开启/关闭唤醒词功能
speech.toggle_wake_word(False)  # 关闭唤醒词功能，直接识别所有语音
speech.toggle_wake_word(True)   # 开启唤醒词功能

# 手动变更监听状态
speech.enable_listening()       # 开启监听
speech.disable_listening()      结束监听
```

### 关键词管理

```python
# 添加关键词及其分类
speech.add_keyword("打开车窗", "window_control")
speech.add_keyword("关闭车窗", "window_control")
speech.add_keyword("打开空调", "ac_control")

# 删除关键词
speech.remove_keyword("打开车窗")

# 添加退出聆听的关键词
speech.add_exit_keyword("退出聆听")

# 删除退出关键词
speech.remove_exit_keyword("再见")

# 获取所有退出关键词
exit_keywords = speech.get_exit_keywords()
```

### 声纹管理

```python
# 注册新的声纹（进入注册模式，需要用户说话）
speech.register_speaker("张三")

# 删除已注册的声纹
speech.delete_speaker("speaker_id")

# 获取所有注册的声纹
speakers = speech.get_registered_speakers()
for speaker_id, info in speakers.items():
    print(f"ID: {speaker_id}, 名称: {info['name']}")

# 获取临时声纹
temp_speakers = speech.get_temp_speakers()

# 将临时声纹升级为正式声纹
speech.promote_temp_speaker("temp_id", "李四")

# 设置最大临时声纹数量
speech.set_max_temp_speakers(5)
```

## 配置说明

语音识别模块会在首次启动时创建默认配置文件，保存在`models/speech/config.json`中。配置项包括：

- `wake_words`: 唤醒词列表
- `wake_words_pinyin`: 唤醒词拼音列表（自动生成）
- `enable_wake_word`: 是否启用唤醒词功能
- `enable_speaker_verification`: 是否启用声纹识别
- `speaker_verification_threshold`: 声纹识别阈值，越低越严格
- `min_enrollment_duration`: 声纹注册最小时长（秒）
- `max_temp_speakers`: 最大临时声纹数量
- `keywords`: 特定关键词及其分类
- `exit_keywords`: 用于退出聆听状态的关键词列表

## 运行演示程序

项目提供了一个完整的演示程序：

```bash
python Modality/demo_speech_recognition.py [options]
```

可用的命令行选项：

- `--no-wake`: 关闭唤醒词功能，直接识别所有语音
- `--add-wake "你好小车"`: 添加自定义唤醒词
- `--register`: 启动后立即进入声纹注册模式
- `--register-name "用户名"`: 注册声纹时使用的用户名
- `--max-temp 5`: 设置临时声纹最大保存数量（默认10个）
- `--debug`: 开启调试模式，显示更多日志信息

## 模型和缓存

模型文件默认保存在以下目录：

- `models/speech/model_cache`: 模型缓存目录
- `models/speech/speaker_db`: 声纹数据库目录
- `models/speech/temp_speakers`: 临时声纹存储目录
- `models/speech/output`: 临时输出目录
