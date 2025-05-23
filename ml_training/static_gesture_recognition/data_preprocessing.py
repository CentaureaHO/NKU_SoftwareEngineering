import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os
import cv2
import mediapipe as mp
import pandas as pd

def normalize_landmarks(landmarks):
    """将关键点标准化到单位框内，以提高稳定性"""
    # 提取坐标
    x_coords = np.array([lm[0] for lm in landmarks])
    y_coords = np.array([lm[1] for lm in landmarks])
    z_coords = np.array([lm[2] for lm in landmarks])
    
    # 确定边界框
    min_x, max_x = np.min(x_coords), np.max(x_coords)
    min_y, max_y = np.min(y_coords), np.max(y_coords)
    min_z, max_z = np.min(z_coords), np.max(z_coords)
    
    # 计算宽度和高度，添加小值防止除以零
    width = max(max_x - min_x, 1e-5)
    height = max(max_y - min_y, 1e-5)
    depth = max(max_z - min_z, 1e-5)
    
    # 归一化到[0,1]范围
    x_normalized = (x_coords - min_x) / width
    y_normalized = (y_coords - min_y) / height
    z_normalized = (z_coords - min_z) / depth
    
    # 合并回landmarks格式
    normalized_landmarks = []
    for i in range(len(landmarks)):
        normalized_landmarks.append([x_normalized[i], y_normalized[i], z_normalized[i]])
    
    return normalized_landmarks

def calculate_angles(landmarks):
    """计算手指间的关键角度，作为特征"""
    angles = []
    
    # 手指基部关节 (指根)
    finger_bases = [1, 5, 9, 13, 17]
    # 手指关节 (中间关节)
    finger_joints = [2, 6, 10, 14, 18]
    # 手指顶部 (指尖)
    finger_tips = [4, 8, 12, 16, 20]
    
    # 手腕位置
    wrist = np.array(landmarks[0])
    
    # 计算手指弯曲角度
    for base, joint, tip in zip(finger_bases, finger_joints, finger_tips):
        # 向量1: 从基部到关节
        v1 = np.array(landmarks[joint]) - np.array(landmarks[base])
        # 向量2: 从关节到指尖
        v2 = np.array(landmarks[tip]) - np.array(landmarks[joint])
        
        # 计算角度 (使用点积)
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)  # 避免数值误差
        angle = np.arccos(cos_angle)
        angles.append(angle)
    
    # 计算指尖之间的相对位置
    for i in range(5):
        for j in range(i+1, 5):
            tip_i = np.array(landmarks[finger_tips[i]])
            tip_j = np.array(landmarks[finger_tips[j]])
            
            # 计算距离
            distance = np.linalg.norm(tip_i - tip_j)
            angles.append(distance)
    
    # 计算拇指与其它指尖的夹角
    thumb_tip = np.array(landmarks[4])
    for i in range(1, 5):
        tip = np.array(landmarks[finger_tips[i]])
        v1 = thumb_tip - wrist
        v2 = tip - wrist
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle)
        angles.append(angle)
    
    return angles

def extract_features(landmarks):
    """从手部关键点提取强健的特征"""
    # 归一化关键点
    normalized_landmarks = normalize_landmarks(landmarks)
    
    # 提取手指角度特征
    angle_features = calculate_angles(normalized_landmarks)
    
    # 展平关键点坐标
    coordinate_features = []
    for lm in normalized_landmarks:
        coordinate_features.extend(lm)
    
    # 合并所有特征
    features = np.concatenate([coordinate_features, angle_features])
    
    return features

def prepare_dataset(csv_file, output_file='hand_features.npz', test_size=0.2):
    """从CSV文件准备数据集"""
    try:
        # 检查文件是否存在
        if not os.path.exists(csv_file):
            print(f"错误: 文件 {csv_file} 不存在!")
            return False
        
        # 读取数据
        print(f"加载数据: {csv_file}")
        df = pd.read_csv(csv_file)
        
        # 提取标签
        labels = df.iloc[:, -1].values
        
        # 准备数据容器
        features_list = []
        
        # 处理每个样本
        for idx in range(len(df)):
            # 提取关键点坐标
            landmarks = []
            for i in range(21):  # 21个关键点
                x = df.iloc[idx, i]
                y = df.iloc[idx, i+21]
                z = df.iloc[idx, i+42]
                landmarks.append([x, y, z])
            
            # 提取特征
            features = extract_features(landmarks)
            features_list.append(features)
        
        # 转换为numpy数组
        X = np.array(features_list)
        y = labels
        
        # 标准化特征
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 分割数据集
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, 
            test_size=test_size, 
            random_state=42,
            stratify=y  # 确保标签分布平衡
        )
        
        # 转换标签为one-hot编码
        num_classes = len(np.unique(y))
        y_train_onehot = np.eye(num_classes)[y_train.astype(int)]
        y_test_onehot = np.eye(num_classes)[y_test.astype(int)]
        
        # 保存处理后的数据
        np.savez(output_file,
                 X_train=X_train,
                 X_test=X_test,
                 y_train=y_train_onehot,
                 y_test=y_test_onehot)
        
        # 保存标准化参数，用于实时预测
        np.save('Modality/models/static_gesture_recognition/model_data/feature_mean.npy', scaler.mean_)
        np.save('Modality/models/static_gesture_recognition/model_data/feature_scale.npy', scaler.scale_)
        
        print(f"特征维度: {X.shape[1]}")
        print(f"样本数量: {X.shape[0]}")
        print(f"类别数量: {num_classes}")
        print(f"数据已保存到 {output_file}")
        
        return True
    
    except Exception as e:
        print(f"准备数据集时出错: {e}")
        return False

def extract_realtime_features(landmarks):
    """从MediaPipe实时检测的关键点提取特征"""
    # 转换MediaPipe格式到我们的格式
    landmark_list = []
    for lm in landmarks:
        landmark_list.append([lm.x, lm.y, lm.z])
    
    # 提取特征
    return extract_features(landmark_list)

# 测试代码
if __name__ == "__main__":
    # 创建模型数据目录
    os.makedirs('Modality/models/static_gesture_recognition/model_data', exist_ok=True)
    
    # 准备训练数据
    if os.path.exists('Dataset/static_gesture_recognition/augmented_gesture_data.csv'):
        print("使用增强后的数据集...")
        prepare_dataset('Dataset/static_gesture_recognition/augmented_gesture_data.csv', 'Modality/models/static_gesture_recognition/model_data/hand_features.npz')
    else:
        print("使用原始数据集...")
        prepare_dataset('Dataset/static_gesture_recognition/gesture_data.csv', 'Modality/models/static_gesture_recognition/model_data/hand_features.npz') 