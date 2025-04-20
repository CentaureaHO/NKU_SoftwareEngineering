# 手势识别模态使用说明

本文档介绍Modality框架中的手势识别模态(GestureTracker)的使用方法。

## 功能介绍

手势识别模态能够识别以下手势:

- 数字0-9的手势
- "ignore"类别（非特定手势或手势不清晰）

## 安装依赖

确保安装了以下依赖项:

```bash
pip install -r requirements.txt
```

## 模型文件

手势识别模态需要以下模型文件:

- `model_data/gesture_model.h5`: 手势识别模型
- `model_data/feature_mean.npy`: 特征均值文件
- `model_data/feature_scale.npy`: 特征缩放文件

如果您从gesture_recognition项目集成，可以直接复制模型文件:

```bash
mkdir -p model_data
cp ../gesture_recognition/model_data/gesture_model.h5 model_data/
cp ../gesture_recognition/model_data/feature_mean.npy model_data/
cp ../gesture_recognition/model_data/feature_scale.npy model_data/
```

## 基本用法

以下是一个基本的使用示例:

```python
from Modality import ModalityManager, GestureTracker

# 创建模态管理器
manager = ModalityManager()

# 创建手势识别模态
gesture_tracker = GestureTracker(
    name="gesture_tracker",   # 模态名称
    source=0,                 # 摄像头ID
    width=640,                # 图像宽度
    height=480,               # 图像高度
    model_path='model_data/gesture_model.h5',          # 模型路径
    feature_mean_path='model_data/feature_mean.npy',   # 特征均值路径
    feature_scale_path='model_data/feature_scale.npy', # 特征缩放路径
    confidence_threshold=0.75,   # 置信度阈值
    stability_threshold=0.7,     # 稳定性阈值
    min_history_size=5,          # 最小历史记录大小
    debug=False                  # 调试模式
)

# 注册模态
manager.register_modality(gesture_tracker)

# 启动模态
manager.start_modality("gesture_tracker")

# 主循环
while True:
    # 更新所有模态
    states = manager.update_all()
  
    # 获取手势状态
    if "gesture_tracker" in states:
        gesture_state = states["gesture_tracker"]
      
        # 获取手势信息
        gesture_info = gesture_state.detections["gesture"]
      
        if gesture_info["detected"]:
            # 获取手势ID和名称
            gesture_id = gesture_info["id"]
            gesture_name = gesture_info["name"]
            confidence = gesture_info["confidence"]
          
            print(f"检测到手势: {gesture_name}, ID: {gesture_id}, 置信度: {confidence:.2f}")
  
    # 退出条件...

# 关闭所有模态
manager.shutdown_all()
```

## 演示程序

可以运行演示程序来测试手势识别功能:

```bash
python demo_gesture_recognition.py
```

### 命令行参数

演示程序支持以下命令行参数:

- `--camera`: 摄像头ID (默认: 0)
- `--width`: 图像宽度 (默认: 640)
- `--height`: 图像高度 (默认: 480)
- `--model`: 模型路径 (默认: model_data/gesture_model.h5)
- `--feature-mean`: 特征均值文件路径 (默认: model_data/feature_mean.npy)
- `--feature-scale`: 特征缩放文件路径 (默认: model_data/feature_scale.npy)
- `--debug`: 开启调试模式

例如:

```bash
python demo_gesture_recognition.py --camera 1 --width 800 --height 600 --debug
```

## 手势状态结构

手势检测结果通过 `GestureState`对象返回，结构如下:

```
{
    "gesture": {
        "id": 数字ID (0-9或10表示"ignore"),
        "name": 手势名称,
        "confidence": 置信度 (0.0-1.0),
        "detected": 是否检测到手势,
        "landmarks": 手部关键点列表,
        "all_probabilities": 所有手势类别的概率列表,
        "stability": 手势稳定性 (0.0-1.0)
    }
}
```

## 注意事项

1. 确保有足够的光线，使手部清晰可见
2. 手部应当完全在摄像头视野范围内
3. 手势应当清晰，并保持稳定以提高识别率
4. 如果识别不准确，可以尝试调整 `confidence_threshold`和 `stability_threshold`参数

## 故障排除

如果遇到问题:

1. 确认模型文件存在且路径正确
2. 检查摄像头ID是否正确
3. 尝试开启调试模式获取更多信息: `--debug`
4. 检查日志文件: `gesture_tracker.log`
