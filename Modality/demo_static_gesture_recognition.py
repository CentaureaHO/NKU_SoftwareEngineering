"""
手势识别模态演示程序

这个示例展示了如何使用Modality库中的手势识别模态来检测和识别手势
"""

import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import cv2
import time
import numpy as np
import argparse
from PIL import Image, ImageDraw, ImageFont

from Modality import ModalityManager, GestureTracker

# 字体设置
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.7
FONT_THICKNESS = 2

# 颜色设置
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)
COLOR_BLUE = (255, 0, 0)
COLOR_YELLOW = (0, 255, 255)
COLOR_WHITE = (255, 255, 255)

# 中文字体支持
def put_chinese_text(img, text, position, font_size=30, color=(0, 255, 0)):
    """在图片上添加中文文本"""
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    
    try:
        # 尝试使用系统中文字体
        fontpath = os.path.join(os.environ.get('SYSTEMROOT', 'C:\\Windows'), 'Fonts', 'simhei.ttf')
        if os.path.exists(fontpath):
            font = ImageFont.truetype(fontpath, font_size)
        else:
            # 尝试其他常见字体
            try:
                font = ImageFont.truetype("simhei.ttf", font_size)
            except IOError:
                try:
                    font = ImageFont.truetype("simsun.ttc", font_size)
                except IOError:
                    font = ImageFont.load_default()
                    print("警告：未能加载中文字体，将使用默认字体")
    except Exception as e:
        print(f"加载字体出错：{e}")
        font = ImageFont.load_default()
    
    draw.text(position, text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='手势识别模态演示')
    
    # 摄像头相关参数
    parser.add_argument('--camera', type=int, default=0, help='摄像头设备ID (默认: 0)')
    parser.add_argument('--width', type=int, default=640, help='摄像头图像宽度 (默认: 640)')
    parser.add_argument('--height', type=int, default=480, help='摄像头图像高度 (默认: 480)')
    
    # 模型相关参数
    parser.add_argument('--model', type=str, default='Modality/models/static_gesture_recognition/model_output/gesture_model.h5', help='手势识别模型路径 (默认: Modality/models/static_gesture_recognition/model_output/gesture_model.h5)')
    parser.add_argument('--feature-mean', type=str, default='Modality/models/static_gesture_recognition/model_data/feature_mean.npy', help='特征均值文件路径 (默认: Modality/models/static_gesture_recognition/model_data/feature_mean.npy)')
    parser.add_argument('--feature-scale', type=str, default='Modality/models/static_gesture_recognition/model_data/feature_scale.npy', help='特征缩放文件路径 (默认: Modality/models/static_gesture_recognition/model_data/feature_scale.npy)')
    
    # 识别相关参数
    parser.add_argument('--confidence', type=float, default=0.75, help='手势识别置信度阈值 (默认: 0.75)')
    parser.add_argument('--stability', type=float, default=0.7, help='手势稳定性阈值 (默认: 0.7)')
    
    # 调试相关参数
    parser.add_argument('--debug', action='store_true', help='开启调试模式')
    
    return parser.parse_args()

# 手势名称中文映射
GESTURE_NAMES_CN = {
    "0": "数字0", "1": "数字1", "2": "数字2", "3": "数字3", "4": "数字4",
    "5": "数字5", "6": "数字6", "7": "数字7", "8": "数字8", "9": "数字9",
    "ignore": "忽略", "unknown": "未知"
}

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_arguments()
    
    # 如果模型目录不存在，创建它
    model_dir = os.path.dirname(args.model)
    if model_dir and not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)
    
    # 检查模型文件是否存在
    if not os.path.exists(args.model) or not os.path.exists(args.feature_mean) or not os.path.exists(args.feature_scale):
        print(f"错误: 模型文件不存在。请确保以下文件存在:")
        print(f"  - {args.model}")
        print(f"  - {args.feature_mean}")
        print(f"  - {args.feature_scale}")
        print("您可以通过命令行参数指定正确的模型文件路径:")
        print("  --model 模型文件路径")
        print("  --feature-mean 特征均值文件路径")
        print("  --feature-scale 特征缩放文件路径")
        return
    
    # 设置环境变量
    if args.debug:
        os.environ['MODALITY_DEBUG'] = '1'
    
    print(f"初始化手势识别模态...")
    print(f"  - 摄像头: {args.camera}")
    print(f"  - 分辨率: {args.width}x{args.height}")
    print(f"  - 模型: {args.model}")
    print(f"  - 置信度阈值: {args.confidence}")
    print(f"  - 稳定性阈值: {args.stability}")
    
    # 创建模态管理器
    manager = ModalityManager()
    
    # 创建手势识别模态
    static_gesture_tracker = GestureTracker(
        name="static_gesture_tracker",
        source=args.camera,
        width=args.width,
        height=args.height,
        model_path=args.model,
        feature_mean_path=args.feature_mean,
        feature_scale_path=args.feature_scale,
        confidence_threshold=args.confidence,
        stability_threshold=args.stability,
        debug=args.debug
    )
    
    # 注册模态
    manager.register_modality(static_gesture_tracker)
    
    # 启动模态
    result = manager.start_modality("static_gesture_tracker")
    if result != 0:
        print(f"错误: 启动手势识别模态失败，错误码: {result}")
        return
    
    print("手势识别模态启动成功")
    print("按ESC键退出")
    
    try:
        # 主循环
        last_time = time.time()
        while True:
            # 计算FPS
            current_time = time.time()
            fps = 1 / (current_time - last_time) if (current_time - last_time) > 0 else 0
            last_time = current_time
            
            # 更新所有模态
            states = manager.update_all()
            
            # 获取手势状态
            if "static_gesture_tracker" in states:
                gesture_state = states["static_gesture_tracker"]
                frame = gesture_state.frame
                
                if frame is not None:
                    # 获取手势检测结果
                    gesture_info = gesture_state.detections["gesture"]
                    
                    # 在界面上显示信息
                    # FPS
                    frame = put_chinese_text(frame, f"FPS: {int(fps)}", (10, 30), font_size=24, color=COLOR_GREEN)
                    
                    # 手势识别结果
                    if gesture_info["detected"]:
                        # 显示手势名称
                        gesture_name = gesture_info["name"]
                        confidence = gesture_info["confidence"]
                        stability = gesture_info["stability"]
                        
                        # 获取中文手势名称
                        gesture_name_cn = GESTURE_NAMES_CN.get(gesture_name, "未知")
                        
                        # 选择颜色
                        color = COLOR_GREEN if gesture_name != "ignore" else COLOR_YELLOW
                        
                        # 显示手势名称和置信度
                        text = f"手势: {gesture_name_cn} ({confidence:.2f})"
                        frame = put_chinese_text(frame, text, (10, 70), font_size=28, color=color)
                        
                        # 显示稳定性
                        stability_text = f"稳定性: {stability:.2f}"
                        frame = put_chinese_text(frame, stability_text, (10, 110), font_size=24, color=color)
                        
                        # 如果有置信度信息，显示所有手势的置信度
                        if "all_probabilities" in gesture_info and gesture_info["all_probabilities"]:
                            probabilities = gesture_info["all_probabilities"]
                            y_offset = 150
                            
                            # 只显示置信度最高的几个手势
                            top_indices = np.argsort(probabilities)[-5:][::-1]
                            
                            for idx in top_indices:
                                prob = probabilities[idx]
                                # 获取中文手势名称
                                idx_gesture_name = str(idx) if idx < 10 else "ignore"
                                idx_gesture_name_cn = GESTURE_NAMES_CN.get(idx_gesture_name, "未知")
                                
                                # 如果是当前手势，用不同颜色显示
                                prob_color = COLOR_RED if idx == gesture_info["id"] else COLOR_BLUE
                                
                                prob_text = f"{idx_gesture_name_cn}: {prob:.3f}"
                                frame = put_chinese_text(frame, prob_text, (10, y_offset), font_size=22, color=prob_color)
                                y_offset += 30
                    else:
                        # 未检测到手势
                        frame = put_chinese_text(frame, "未检测到手势", (10, 70), font_size=28, color=COLOR_RED)
                    
                    # 显示使用指导
                    guide = "按ESC键退出"
                    frame = put_chinese_text(frame, guide, (10, frame.shape[0] - 30), font_size=22, color=COLOR_WHITE)
                    
                    # 显示图像
                    cv2.imshow("手势识别演示", frame)
                
                # 按ESC键退出
                key = cv2.waitKey(1)
                if key == 27:  # ESC
                    break
            else:
                print("未收到手势识别状态更新")
                time.sleep(0.01)
    
    except KeyboardInterrupt:
        print("程序被用户中断")
    except Exception as e:
        print(f"发生错误: {str(e)}")
    finally:
        # 关闭所有模态
        manager.shutdown_all()
        cv2.destroyAllWindows()
        print("程序已退出")

if __name__ == "__main__":
    main()
    