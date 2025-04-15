import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model, save_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import time

# 全局设置
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)

# 设置TensorFlow内存增长
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

def build_model(input_shape, num_classes):
    """构建更加稳定和高效的手势识别模型"""
    model = Sequential([
        # 输入层
        Dense(256, input_shape=(input_shape,), 
              kernel_regularizer=l2(0.0005),
              kernel_initializer='he_normal'),
        BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        Dropout(0.3),
        
        # 隐藏层1
        Dense(256, kernel_regularizer=l2(0.0005),
              kernel_initializer='he_normal'),
        BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        Dropout(0.3),
        
        # 隐藏层2
        Dense(128, kernel_regularizer=l2(0.0005),
              kernel_initializer='he_normal'),
        BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        Dropout(0.3),
        
        # 输出层
        Dense(num_classes, activation='softmax')
    ])
    
    # 编译模型
    optimizer = Adam(learning_rate=0.001)
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def train_model(data_file, model_dir='model_data', epochs=100, batch_size=64):
    """训练改进的手势识别模型"""
    
    # 创建模型目录
    os.makedirs(model_dir, exist_ok=True)
    
    # 加载数据
    print(f"加载数据: {data_file}")
    try:
        data = np.load(data_file)
        X_train = data['X_train']
        X_test = data['X_test']
        y_train = data['y_train']
        y_test = data['y_test']
    except Exception as e:
        print(f"加载数据失败: {e}")
        return None
    
    input_shape = X_train.shape[1]
    num_classes = y_train.shape[1]
    
    print(f"构建模型 (输入维度: {input_shape}, 输出类别数: {num_classes})")
    model = build_model(input_shape, num_classes)
    model.summary()
    
    # 回调函数
    callbacks = [
        # 提前停止 - 避免过拟合
        EarlyStopping(
            monitor='val_accuracy',
            patience=20,
            restore_best_weights=True,
            verbose=1
        ),
        # 保存最佳模型
        ModelCheckpoint(
            filepath=os.path.join(model_dir, 'best_model.h5'),
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        # 学习率调整
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    # 记录开始时间
    start_time = time.time()
    
    # 训练模型
    print("开始训练模型...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    # 训练时间
    training_time = time.time() - start_time
    print(f"训练完成，耗时: {training_time:.2f} 秒")
    
    # 评估模型
    print("评估模型...")
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print(f"测试集准确率: {test_acc:.4f}")
    
    # 保存最终模型
    final_model_path = os.path.join(model_dir, 'gesture_model.h5')
    model.save(final_model_path)
    print(f"模型已保存到: {final_model_path}")
    
    # 获取预测结果
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true_classes = np.argmax(y_test, axis=1)
    
    # 绘制混淆矩阵
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_true_classes, y_pred_classes)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('混淆矩阵')
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.savefig(os.path.join(model_dir, 'confusion_matrix.png'))
    
    # 生成分类报告
    report = classification_report(y_true_classes, y_pred_classes)
    print("\n分类报告:")
    print(report)
    
    # 保存分类报告到文件
    with open(os.path.join(model_dir, 'classification_report.txt'), 'w') as f:
        f.write(report)
    
    # 绘制训练历史
    plt.figure(figsize=(12, 5))
    
    # 准确率
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='训练集准确率')
    plt.plot(history.history['val_accuracy'], label='验证集准确率')
    plt.title('模型准确率')
    plt.xlabel('Epoch')
    plt.ylabel('准确率')
    plt.legend()
    
    # 损失
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='训练集损失')
    plt.plot(history.history['val_loss'], label='验证集损失')
    plt.title('模型损失')
    plt.xlabel('Epoch')
    plt.ylabel('损失')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(model_dir, 'training_history.png'))
    
    return model

if __name__ == "__main__":
    # 训练模型
    train_model('model_data/hand_features.npz', model_dir='model_data') 