import os
import shutil
import numpy as np
import sys
import importlib

def create_directory_structure():
    """创建必要的目录结构"""
    # 创建模型数据目录
    os.makedirs('model_data', exist_ok=True)
    print("创建目录: model_data/")

def move_existing_files():
    """移动现有的模型和参数文件到模型数据目录"""
    files_to_move = [
        ('final_gesture_model.h5', 'model_data/backup_model.h5'),
        ('best_model.h5', 'model_data/best_model.h5'),
        ('scaler_mean.npy', 'model_data/backup_mean.npy'),
        ('scaler_scale.npy', 'model_data/backup_scale.npy')
    ]
    
    for src, dst in files_to_move:
        if os.path.exists(src):
            try:
                shutil.copy2(src, dst)
                print(f"已复制: {src} -> {dst}")
            except Exception as e:
                print(f"复制 {src} 到 {dst} 失败: {e}")

def create_dummy_files():
    """创建临时文件，确保程序能够启动"""
    # 如果没有模型文件，创建简单的占位符
    model_file = 'model_data/gesture_model.h5'
    if not (os.path.exists(model_file) or 
            os.path.exists('model_data/backup_model.h5') or 
            os.path.exists('final_gesture_model.h5') or
            os.path.exists('best_model.h5')):
        print("警告: 未找到模型文件，需要先训练模型")
    
    # 创建必要的均值和方差文件（如果不存在）
    if not (os.path.exists('model_data/feature_mean.npy') or os.path.exists('scaler_mean.npy')):
        dummy_mean = np.zeros(63)  # 基本特征维度
        np.save('model_data/feature_mean.npy', dummy_mean)
        print("创建临时均值文件")
    
    if not (os.path.exists('model_data/feature_scale.npy') or os.path.exists('scaler_scale.npy')):
        dummy_scale = np.ones(63)  # 基本特征维度
        np.save('model_data/feature_scale.npy', dummy_scale)
        print("创建临时方差文件")

def check_dependencies():
    """检查必要的依赖项"""
    required_modules = {
        'tensorflow': 'tensorflow',
        'cv2': 'opencv-python',
        'mediapipe': 'mediapipe',
        'numpy': 'numpy',
        'PIL': 'pillow',
        'sklearn': 'scikit-learn'
    }
    
    missing_packages = []
    
    # 尝试导入每个模块
    for module_name, package_name in required_modules.items():
        try:
            # 直接尝试导入模块
            importlib.import_module(module_name)
            print(f"√ 依赖项已满足: {package_name}")
        except ImportError:
            print(f"× 缺少依赖项: {package_name}")
            missing_packages.append(package_name)
    
    if missing_packages:
        print("\n缺少以下必要依赖项:")
        for package in missing_packages:
            print(f"  - {package}")
        print("\n请使用以下命令安装:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    print("\n所有依赖项检查通过")
    return True

def main():
    """主函数"""
    print("开始设置手势识别环境...")
    
    # 创建目录结构
    create_directory_structure()
    
    # 移动现有文件
    move_existing_files()
    
    # 创建临时文件
    create_dummy_files()
    
    # 检查依赖项
    check_dependencies()
    
    print("""
设置完成！

使用说明:
1. 首先收集手势数据:
   python data_collection.py

2. 增强数据集(如果需要):
   python data_augmentation.py

3. 处理数据并训练新模型:
   python simplified_preprocessing.py
   python improved_model.py

4. 运行手势识别:
   python gesture_recognition.py
""")

if __name__ == "__main__":
    main() 