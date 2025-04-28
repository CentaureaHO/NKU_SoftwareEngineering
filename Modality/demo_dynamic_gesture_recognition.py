"""
动态手势识别模态演示程序

这个示例展示了如何使用Modality库中的动态手势识别模态来检测和识别手势动作，
并显示前五个预测结果及其置信度分数。
"""

import sys
import os
import argparse
import cv2
import numpy as np
import time
from PIL import Image, ImageDraw, ImageFont

def put_chinese_text(img, text, position, font_size=20, color=(255, 255, 255), thickness=1):
    """在OpenCV图像上绘制中文文本"""
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    try:
        # 优先尝试微软雅黑或其他常见中文字体
        font_paths = [
            "C:/Windows/Fonts/msyh.ttc",         # Windows 微软雅黑
            "C:/Windows/Fonts/simhei.ttf",       # Windows 黑体
            "C:/Windows/Fonts/simsun.ttc",       # Windows 宋体
            "/System/Library/Fonts/PingFang.ttc" # macOS
        ]
        
        font = None
        for path in font_paths:
            if os.path.exists(path):
                font = ImageFont.truetype(path, font_size)
                break
                
        if font is None:
            # 如果找不到上述字体，使用默认字体
            font = ImageFont.load_default()
    except IOError:
        # 如果加载失败，使用默认字体
        font = ImageFont.load_default()
    
    # 在PIL图像上绘制文本
    draw = ImageDraw.Draw(img_pil)
    draw.text(position, text, font=font, fill=color)
    
    # 转换回OpenCV格式
    img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    return img_cv

def main():
    # 获取当前脚本所在目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 获取项目根目录
    root_dir = os.path.dirname(current_dir)
    # 将项目根目录添加到路径中
    sys.path.append(root_dir)
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='动态手势识别演示')
    parser.add_argument('--camera', type=int, default=0, help='摄像头索引 (默认: 0)')
    parser.add_argument('--model', type=str, default='Modality/models/dynamic_gesture_recognition/jester_conv_example/model_best.pth.tar', 
                        help='模型路径')
    parser.add_argument('--model_class', type=str, default='Modality/models/dynamic_gesture_recognition/model.py', 
                        help='模型类文件路径')
    parser.add_argument('--labels', type=str, default='Dataset/dynamic_gesture_recognition/annotations/jester-v1-labels.csv', 
                        help='标签文件路径')
    parser.add_argument('--quick_test', action='store_true', help='使用快速测试模型和配置')
    parser.add_argument('--confidence', type=float, default=0.65, help='置信度阈值')
    parser.add_argument('--motion_threshold', type=float, default=500.0, help='运动检测阈值')
    parser.add_argument('--debug', action='store_true', help='启用调试模式')
    parser.add_argument('--input_width', type=int, default=176, help='输入图像宽度')
    parser.add_argument('--input_height', type=int, default=100, help='输入图像高度')
    args = parser.parse_args()
    
    # 如果选择快速测试模式，使用快速测试模型和配置
    if args.quick_test:
        args.model = 'Modality/models/dynamic_gesture_recognition/jester_conv_example/model_best.pth.tar'
        args.labels = 'Dataset/dynamic_gesture_recognition/annotations/jester-v1-labels-quick-testing.csv'
        print("使用快速测试模型和配置")
    else:
        print("使用完整训练模型和配置")

    # 检查模型文件是否存在
    if not os.path.exists(args.model):
        print(f"错误: 模型文件不存在: {args.model}")
        print("您需要先训练模型或者提供正确的模型路径。")
        print("是否继续? [y/N]")
        choice = input().strip().lower()
        if choice != 'y':
            return
    
    if not os.path.exists(args.model_class):
        print(f"警告: 模型类文件不存在: {args.model_class}")
        print("请确保模型类文件存在。")
        return
    
    if not os.path.exists(args.labels):
        print(f"警告: 标签文件不存在: {args.labels}")
        print("请确保标签文件存在。")
        return
    
    # 导入动态手势跟踪器
    from Modality.visual import DynamicGestureTracker
    
    # 创建跟踪器
    print(f"初始化动态手势识别模态...")
    print(f"  - 摄像头: {args.camera}")
    print(f"  - 分辨率: 640x480 (捕获), {args.input_width}x{args.input_height} (处理)")
    print(f"  - 模型: {args.model}")
    print(f"  - 模型类: {args.model_class}")
    print(f"  - 标签: {args.labels}")
    print(f"  - 置信度阈值: {args.confidence}")
    print(f"  - 运动检测阈值: {args.motion_threshold}")
    
    tracker = DynamicGestureTracker(
        source=args.camera,
        model_path=args.model,
        model_class_path=args.model_class,
        label_path=args.labels,
        confidence_threshold=args.confidence,
        motion_threshold=args.motion_threshold,
        input_width=args.input_width,
        input_height=args.input_height,
        debug=True  # 始终开启调试模式以获取前五个预测
    )
    
    # 初始化跟踪器
    result = tracker.initialize()
    if result != 0:
        print(f"错误: 启动动态手势识别模态失败，错误码: {result}")
        return
    
    print(f"  - 历史设置: 大小={tracker.history_size}, 共识={tracker.min_consensus}, 冷却={tracker.prediction_cooldown}s")
    
    # 启动跟踪器
    tracker.start()
    print("动态手势识别模态已启动")
    
    print("\n按键指南:")
    print("- 'q': 退出程序")
    print("- 'd': 切换调试模式")
    print("- 's': 增加平滑度")
    print("- 'a': 减少平滑度")
    print("- '+': 增加运动阈值")
    print("- '-': 减少运动阈值")
    
    # 创建窗口
    cv2.namedWindow('动态手势识别系统', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('动态手势识别系统', 800, 600)
    
    # FPS计算
    fps_counter = 0
    fps_start_time = time.time()
    fps = 0
    
    # 调试模式
    debug_mode = args.debug
    
    # 用于持续显示的变量
    display_frames = 0
    display_duration = 30  # 显示结果的持续时间
    
    while True:
        # 计时开始
        start_time = time.time()
        
        # 更新跟踪器状态
        state = tracker.update()
        if state is None:
            print("无法获取状态，跳过本帧")
            time.sleep(0.01)
            continue
        
        # 获取原始帧
        frame = state.frame
        if frame is None:
            print("无法获取帧，跳过本帧")
            time.sleep(0.01)
            continue
        
        # 创建显示界面
        display_frame = cv2.resize(frame, (800, 600))
        
        # 更新FPS计数
        fps_counter += 1
        if time.time() - fps_start_time > 1.0:
            fps = fps_counter / (time.time() - fps_start_time)
            fps_counter = 0
            fps_start_time = time.time()
        
        # 添加半透明的信息面板
        info_panel = display_frame.copy()
        panel_height = 220  # 增大面板高度，以容纳前五个预测
        cv2.rectangle(info_panel, (0, 0), (800, panel_height), (0, 0, 0), -1)
        alpha = 0.7
        display_frame = cv2.addWeighted(info_panel, alpha, display_frame, 1 - alpha, 0)
        
        # 显示设置
        display_frame = put_chinese_text(
            display_frame, 
            f"平滑: {tracker.history_size}/{tracker.min_consensus}, 冷却: {tracker.prediction_cooldown:.1f}s, 运动阈值: {tracker.motion_threshold:.1f}", 
            (20, 30), 
            font_size=15,
            color=(150, 150, 150)
        )
        
        # 显示FPS
        display_frame = put_chinese_text(
            display_frame, 
            f"FPS: {fps:.1f}", 
            (700, 30), 
            font_size=15,
            color=(255, 255, 255)
        )
        
        # 显示运动级别
        motion_level = state.detections.get("gesture", {}).get("motion_level", 0)
        display_frame = put_chinese_text(
            display_frame, 
            f"当前运动量: {motion_level:.1f}", 
            (550, 60), 
            font_size=15,
            color=(150, 150, 150)
        )
        
        # 获取预测结果
        prediction = state.detections.get("gesture", {}).get("name", "等待手势...")
        confidence = state.detections.get("gesture", {}).get("confidence", 0.0)
        detected = state.detections.get("gesture", {}).get("detected", False)
        
        # 如果检测到手势，重置显示计数器
        if detected and prediction != "等待手势...":
            display_frames = display_duration
        
        # 显示预测结果
        if display_frames > 0 or (prediction != "等待手势..." and detected):
            # 根据置信度设置不同颜色
            if confidence > 0.7:
                color = (0, 255, 0)  # 绿色表示高置信度
            elif confidence > 0.5:
                color = (0, 255, 255)  # 黄色表示中等置信度
            else:
                color = (0, 165, 255)  # 橙色表示低置信度
            
            # 显示标题
            display_frame = put_chinese_text(
                display_frame, 
                "检测到的手势:", 
                (20, 70), 
                font_size=24,
                color=(255, 255, 255)
            )
            
            # 显示预测
            display_frame = put_chinese_text(
                display_frame, 
                f"{prediction}", 
                (20, 100), 
                font_size=24,
                color=color
            )
            
            display_frames -= 1
        else:
            # 如果没有活动预测，显示等待消息
            display_frame = put_chinese_text(
                display_frame, 
                "等待手势...", 
                (20, 100), 
                font_size=24,
                color=(200, 200, 200)
            )
        
        # 始终显示前五个预测结果及其置信度
        top_predictions = state.detections.get("gesture", {}).get("top_predictions", [])
        if top_predictions:
            # 显示前五个预测的标题
            display_frame = put_chinese_text(
                display_frame, 
                "前五个预测结果:", 
                (300, 70), 
                font_size=22,
                color=(255, 255, 255)
            )
            
            # 显示前五个预测及其置信度
            for i, (label, conf) in enumerate(top_predictions):
                # 根据置信度设置颜色
                if conf > 0.7:
                    pred_color = (0, 255, 0)  # 绿色
                elif conf > 0.4:
                    pred_color = (0, 255, 255)  # 黄色
                elif conf > 0.2:
                    pred_color = (0, 165, 255)  # 橙色
                else:
                    pred_color = (150, 150, 150)  # 灰色
                
                display_frame = put_chinese_text(
                    display_frame, 
                    f"{i+1}. {label}: {conf:.3f}", 
                    (320, 100 + i*24), 
                    font_size=20,
                    color=pred_color
                )
        
        # 如果开启调试模式，显示更多信息
        if debug_mode:
            # 显示历史记录
            if hasattr(tracker, 'prediction_history') and tracker.prediction_history:
                hist_y = 240
                hist_text = " -> ".join(list(tracker.prediction_history)[-5:])
                display_frame = put_chinese_text(
                    display_frame, 
                    f"最近预测: {hist_text}", 
                    (30, hist_y), 
                    font_size=16,
                    color=(200, 200, 200)
                )
        
        # 在显示画面上绘制一个矩形框，指示模型实际处理的区域
        display_h, display_w = display_frame.shape[:2]
        
        # 将框扩大到整个摄像头视野，使用176x100的比例
        input_ratio_w = 176 / display_w
        input_ratio_h = 100 / display_h
        
        # 使用整个屏幕作为输入区域
        box_x1 = 0
        box_y1 = 0
        box_x2 = display_w
        box_y2 = display_h
        
        cv2.rectangle(display_frame, (box_x1, box_y1), (box_x2, box_y2), (0, 255, 0), 2)
        display_frame = put_chinese_text(
            display_frame, 
            "全屏手势检测区域", 
            (box_x1 + 10, box_y1 + 30), 
            font_size=16,
            color=(0, 255, 0)
        )
        
        # 显示帧
        cv2.imshow('动态手势识别系统', display_frame)
        
        # 键盘控制
        key = cv2.waitKey(1)
        if key == ord('q'):  # 按q退出
            break
        elif key == ord('d'):  # 按d切换调试模式
            debug_mode = not debug_mode
            print(f"调试模式: {'开启' if debug_mode else '关闭'}")
        elif key == ord('s'):  # 按s增加平滑度
            history_size = min(tracker.history_size + 1, 10)
            min_consensus = min(tracker.min_consensus + 1, history_size)
            prediction_cooldown = min(tracker.prediction_cooldown + 0.2, 3.0)
            tracker.set_parameters(
                history_size=history_size,
                min_consensus=min_consensus,
                prediction_cooldown=prediction_cooldown
            )
            print(f"增加平滑度: 历史大小={history_size}, 所需共识={min_consensus}, 冷却时间={prediction_cooldown:.1f}秒")
        elif key == ord('a'):  # 按a减少平滑度
            history_size = max(tracker.history_size - 1, 3)
            min_consensus = max(tracker.min_consensus - 1, 1)
            prediction_cooldown = max(tracker.prediction_cooldown - 0.2, 0.2)
            tracker.set_parameters(
                history_size=history_size,
                min_consensus=min_consensus,
                prediction_cooldown=prediction_cooldown
            )
            print(f"减少平滑度: 历史大小={history_size}, 所需共识={min_consensus}, 冷却时间={prediction_cooldown:.1f}秒")
        elif key == ord('+') or key == ord('='):  # 按+增加运动阈值
            motion_threshold = min(tracker.motion_threshold + 50, 2000)
            tracker.set_parameters(motion_threshold=motion_threshold)
            print(f"增加运动阈值: {motion_threshold:.1f}")
        elif key == ord('-') or key == ord('_'):  # 按-减少运动阈值
            motion_threshold = max(tracker.motion_threshold - 50, 100)
            tracker.set_parameters(motion_threshold=motion_threshold)
            print(f"减少运动阈值: {motion_threshold:.1f}")
        
        # 计算帧处理时间
        frame_time = time.time() - start_time
        target_time = 1.0 / 30  # 目标30FPS
        if frame_time < target_time:
            time.sleep(target_time - frame_time)
    
    # 停止并关闭资源
    print("正在停止跟踪器...")
    tracker.stop()
    cv2.destroyAllWindows()
    print("演示结束")

if __name__ == "__main__":
    main() 