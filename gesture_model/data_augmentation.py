import pandas as pd
import numpy as np
import csv
from scipy.spatial.transform import Rotation

def augment_data(input_file, output_file, augmentation_factor=10):
    """
    对手势数据进行增强，生成更多样本
    """
    # 读取原始数据
    df = pd.read_csv(input_file)
    
    # 分离特征和标签
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    
    # 准备输出数据
    augmented_data = []
    
    # 对每个样本进行增强
    for i in range(len(X)):
        # 添加原始样本
        augmented_data.append(np.append(X[i], y[i]))
        
        # 生成增强样本
        for j in range(augmentation_factor - 1):
            # 手势数据格式: [x1,x2,...,x21, y1,y2,...,y21, z1,z2,...,z21]
            augmented_sample = X[i].copy()
            
            # 1. 计算手腕位置作为参考点（第0个关键点）
            wrist_x = augmented_sample[0]
            wrist_y = augmented_sample[21]
            wrist_z = augmented_sample[42]
            
            # 2. 添加微小位移 (整体平移)
            shift_x = np.random.uniform(-0.05, 0.05)
            shift_y = np.random.uniform(-0.05, 0.05)
            shift_z = np.random.uniform(-0.01, 0.01)  # z轴位移较小
            
            # 应用整体位移
            for k in range(0, 21):
                augmented_sample[k] += shift_x        # x坐标
                augmented_sample[21+k] += shift_y     # y坐标
                augmented_sample[42+k] += shift_z     # z坐标
            
            # 3. 添加微小缩放变化 - 相对于手腕
            scale = np.random.uniform(0.9, 1.1)
            
            # 应用缩放（相对于手腕）
            for k in range(0, 21):
                # x坐标缩放
                augmented_sample[k] = wrist_x + scale * (augmented_sample[k] - wrist_x)
                # y坐标缩放
                augmented_sample[21+k] = wrist_y + scale * (augmented_sample[21+k] - wrist_y)
                # z坐标缩放
                augmented_sample[42+k] = wrist_z + scale * (augmented_sample[42+k] - wrist_z)

            # 4. 添加旋转变换 - 使用手腕作为旋转中心
            # 为XY平面和XZ平面使用不同的旋转角度
            xy_rotation = np.random.uniform(-20, 20) * (np.pi/180) 
            xz_rotation = np.random.uniform(-10, 10) * (np.pi/180)
            
            # XY平面旋转（最常见的手部旋转）
            cos_theta_xy = np.cos(xy_rotation)
            sin_theta_xy = np.sin(xy_rotation)
            
            # XZ平面旋转（模拟手腕前后翻转）
            cos_theta_xz = np.cos(xz_rotation)
            sin_theta_xz = np.sin(xz_rotation)
            
            for k in range(0, 21):
                # 计算相对于手腕的位置
                rel_x = augmented_sample[k] - wrist_x
                rel_y = augmented_sample[21+k] - wrist_y
                rel_z = augmented_sample[42+k] - wrist_z
                
                # 应用XY平面旋转
                new_x = rel_x * cos_theta_xy - rel_y * sin_theta_xy
                new_y = rel_x * sin_theta_xy + rel_y * cos_theta_xy
                
                # 应用XZ平面旋转（仅对部分样本）
                if np.random.random() < 0.5:  # 50%概率应用XZ旋转
                    new_x, new_z = new_x * cos_theta_xz - rel_z * sin_theta_xz, new_x * sin_theta_xz + rel_z * cos_theta_xz
                else:
                    new_z = rel_z
                
                # 更新坐标（加回手腕位置）
                augmented_sample[k] = wrist_x + new_x
                augmented_sample[21+k] = wrist_y + new_y
                augmented_sample[42+k] = wrist_z + new_z
            
            # 5. 添加高斯噪声
            noise_scale = np.random.uniform(0.001, 0.005)
            noise = np.random.normal(0, noise_scale, augmented_sample.shape)
            augmented_sample += noise
            
            # 6. 随机手指抖动（模拟手部微动）
            if np.random.random() < 0.3:  # 30%概率添加手指抖动
                finger_indices = list(range(1, 21))  # 排除手腕
                num_fingers_to_jitter = np.random.randint(1, 5)  # 随机选择1-4个手指
                fingers_to_jitter = np.random.choice(finger_indices, num_fingers_to_jitter, replace=False)
                
                for finger_idx in fingers_to_jitter:
                    jitter_x = np.random.uniform(-0.01, 0.01)
                    jitter_y = np.random.uniform(-0.01, 0.01)
                    jitter_z = np.random.uniform(-0.005, 0.005)
                    
                    augmented_sample[finger_idx] += jitter_x
                    augmented_sample[21 + finger_idx] += jitter_y
                    augmented_sample[42 + finger_idx] += jitter_z
            
            # 保存增强样本和标签
            augmented_data.append(np.append(augmented_sample, y[i]))
    
    # 写入增强后的数据
    with open(output_file, 'w', newline='') as f:
        writer = csv.writer(f)
        # 写入表头
        header = [f'{axis}{i}' for i in range(21) for axis in ['x', 'y', 'z']] + ['label']
        writer.writerow(header)
        # 写入数据
        writer.writerows(augmented_data)
    
    print(f"原始数据: {len(X)} 样本")
    print(f"增强后数据: {len(augmented_data)} 样本")

# 使用示例
if __name__ == "__main__":
    augment_data('Dataset/gesture_data.csv', 'Dataset/augmented_gesture_data.csv', augmentation_factor=10)