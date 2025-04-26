import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
import argparse
import datetime
import glob
import pickle

np.random.seed(42)
tf.random.set_seed(42)

DEFAULT_DATASET_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "Dataset", "headpose")
DEFAULT_MODEL_OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
DEFAULT_WINDOW_SIZE = 15  # 15帧 (约0.5秒 @30FPS)
DEFAULT_STRIDE = 5
DEFAULT_EPOCHS = 50
DEFAULT_BATCH_SIZE = 32
DEFAULT_TEST_SIZE = 0.2
DEFAULT_VAL_SIZE = 0.1

os.makedirs(DEFAULT_MODEL_OUTPUT_DIR, exist_ok=True)

GESTURE_MAPPING = {
    "stationary": 0,
    "nodding": 1,
    "shaking": 2,
    "other": 3
}

INVERSE_GESTURE_MAPPING = {v: k for k, v in GESTURE_MAPPING.items()}

class HeadGestureDataProcessor:
    def __init__(self, dataset_dir, window_size=DEFAULT_WINDOW_SIZE, 
                 stride=DEFAULT_STRIDE, test_size=DEFAULT_TEST_SIZE, 
                 val_size=DEFAULT_VAL_SIZE):
        self.dataset_dir = dataset_dir
        self.window_size = window_size
        self.stride = stride
        self.test_size = test_size
        self.val_size = val_size
        self.all_data_files = []
        self.scaler = StandardScaler()
        self.feature_dim = None

        if not os.path.exists(self.dataset_dir):
            raise ValueError(f"数据集目录不存在: {self.dataset_dir}")

    def _extract_features_from_frame(self, frame):
        if not frame.get("detected", False):
            return None

        if "key_data" not in frame or "raw_data" not in frame:
            return None

        key_data = frame["key_data"]
        
        required_fields = ["pitch", "yaw", "roll", "nose_chin_distance", "left_cheek_width", "right_cheek_width"]
        for field in required_fields:
            if field not in key_data:
                return None
        
        try:
            basic_features = [
                float(key_data["pitch"]),
                float(key_data["yaw"]),
                float(key_data["roll"]),
                float(key_data["nose_chin_distance"]),
                float(key_data["left_cheek_width"]),
                float(key_data["right_cheek_width"])
            ]
            
            if len(frame["raw_data"]) > 0:
                coords = []
                for point in frame["raw_data"]:
                    if "x_px" in point and "y_px" in point:
                        coords.append([point["x_px"], point["y_px"]])
                
                if coords:
                    coords = np.array(coords)
                    
                    x_min, y_min = np.min(coords, axis=0)
                    x_max, y_max = np.max(coords, axis=0)
                    box_width = max(x_max - x_min, 1)
                    box_height = max(y_max - y_min, 1)
                    box_diagonal = np.sqrt(box_width**2 + box_height**2)
                    
                    aspect_ratio = box_width / box_height
                    
                    normalized_features = [
                        aspect_ratio,
                        box_width / box_diagonal,
                        box_height / box_diagonal,
                        
                        float(key_data["pitch"]) / box_diagonal,
                        float(key_data["yaw"]) / box_diagonal,
                        float(key_data["roll"]) / box_diagonal,
                        
                        float(key_data["nose_chin_distance"]) / box_height,
                        float(key_data["left_cheek_width"]) / box_width,
                        float(key_data["right_cheek_width"]) / box_width
                    ]
                    
                    combined_features = np.concatenate([basic_features, normalized_features])
                    return combined_features

            return np.array(basic_features, dtype=np.float32)
            
        except (ValueError, TypeError, KeyError) as e:
            print(f"特征提取错误: {str(e)}")
            return None

    def _create_sequences(self, features_list, label):
        sequences = []
        labels = []

        valid_features = [f for f in features_list if f is not None]

        if len(valid_features) < self.window_size:
            return sequences, labels

        for i in range(0, len(valid_features) - self.window_size + 1, self.stride):
            seq = valid_features[i:i + self.window_size]

            diff_seq = []
            for j in range(1, len(seq)):
                combined_features = np.concatenate([
                    seq[j],
                    seq[j] - seq[j-1]
                ])
                diff_seq.append(combined_features)

            sequences.append(np.array(diff_seq))
            labels.append(label)

        return sequences, labels

    def load_and_preprocess_data(self):
        json_files = glob.glob(os.path.join(self.dataset_dir, "*.json"))
        if not json_files:
            raise ValueError(f"未在 {self.dataset_dir} 中找到数据集")

        all_sequences = []
        all_labels = []

        for json_file in json_files:
            try:
                filename = os.path.basename(json_file)
                gesture_type = filename.split('_')[-1].replace('.json', '')

                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)

                frames = data["frames"]
                features_list = []
                for frame in frames:
                    feature = self._extract_features_from_frame(frame)
                    features_list.append(feature)

                sequences, labels = self._create_sequences(
                    features_list, 
                    GESTURE_MAPPING[data["gesture_type"]]
                )

                all_sequences.extend(sequences)
                all_labels.extend(labels)

            except Exception as e:
                print(f"处理文件 {json_file} 时出错: {str(e)}")

        all_sequences = np.array(all_sequences)
        all_labels = np.array(all_labels)

        self.feature_dim = all_sequences.shape[2]

        print(f"创建了 {len(all_sequences)} 个序列样本，每个序列长度为 {self.window_size-1}，特征维度为 {self.feature_dim}")

        shape = all_sequences.shape
        all_sequences_flat = all_sequences.reshape((-1, self.feature_dim))
        all_sequences_flat = self.scaler.fit_transform(all_sequences_flat)
        all_sequences = all_sequences_flat.reshape(shape)

        train_val_data, test_data, train_val_labels, test_labels = train_test_split(
            all_sequences, all_labels, test_size=self.test_size, random_state=42, stratify=all_labels
        )

        train_data, val_data, train_labels, val_labels = train_test_split(
            train_val_data, train_val_labels, 
            test_size=self.val_size/(1-self.test_size),
            random_state=42,
            stratify=train_val_labels
        )

        return train_data, val_data, test_data, train_labels, val_labels, test_labels


def create_head_gesture_model(seq_length, feature_dim, num_classes=4):
    model = keras.Sequential([
        keras.layers.Input(shape=(seq_length, feature_dim)),

        keras.layers.GRU(64, return_sequences=True),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),

        keras.layers.GRU(32),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),

        keras.layers.Dense(16, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

def plot_training_history(history, output_dir):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()

    plt.savefig(os.path.join(output_dir, f"training_history_{timestamp}.png"))
    plt.close()

def plot_confusion_matrix(y_true, y_pred, output_dir):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=[INVERSE_GESTURE_MAPPING[i] for i in range(len(GESTURE_MAPPING))],
        yticklabels=[INVERSE_GESTURE_MAPPING[i] for i in range(len(GESTURE_MAPPING))]
    )
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

    plt.savefig(os.path.join(output_dir, f"confusion_matrix_{timestamp}.png"))
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="GRU Training")

    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET_DIR,
                       help=f"数据集目录 (默认: {DEFAULT_DATASET_DIR})")
    parser.add_argument("--output", type=str, default=DEFAULT_MODEL_OUTPUT_DIR,
                       help=f"模型输出目录 (默认: {DEFAULT_MODEL_OUTPUT_DIR})")
    parser.add_argument("--window", type=int, default=DEFAULT_WINDOW_SIZE,
                       help=f"窗口大小 (默认: {DEFAULT_WINDOW_SIZE})")
    parser.add_argument("--stride", type=int, default=DEFAULT_STRIDE,
                       help=f"滑动步长 (默认: {DEFAULT_STRIDE})")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS,
                       help=f"训练轮数 (默认: {DEFAULT_EPOCHS})")
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH_SIZE,
                       help=f"批次大小 (默认: {DEFAULT_BATCH_SIZE})")
    parser.add_argument("--test_size", type=float, default=DEFAULT_TEST_SIZE,
                       help=f"测试集比例 (默认: {DEFAULT_TEST_SIZE})")
    parser.add_argument("--val_size", type=float, default=DEFAULT_VAL_SIZE,
                       help=f"验证集比例 (默认: {DEFAULT_VAL_SIZE})")

    args = parser.parse_args()

    try:
        data_processor = HeadGestureDataProcessor(
            dataset_dir=args.dataset,
            window_size=args.window,
            stride=args.stride,
            test_size=args.test_size,
            val_size=args.val_size
        )

        train_data, val_data, test_data, train_labels, val_labels, test_labels = data_processor.load_and_preprocess_data()

        model = create_head_gesture_model(
            seq_length=args.window-1,
            feature_dim=data_processor.feature_dim,
            num_classes=len(GESTURE_MAPPING)
        )

        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-5
            )
        ]

        print(f"\n开始训练模型，共 {args.epochs} 轮，批次大小 {args.batch}")
        history = model.fit(
            train_data, train_labels,
            validation_data=(val_data, val_labels),
            epochs=args.epochs,
            batch_size=args.batch,
            callbacks=callbacks,
            verbose=1
        )

        plot_training_history(history, args.output)

        test_loss, test_accuracy = model.evaluate(test_data, test_labels)
        print(f"测试集损失: {test_loss:.4f}")
        print(f"测试集准确率: {test_accuracy:.4f}")

        y_pred = np.argmax(model.predict(test_data), axis=1)

        plot_confusion_matrix(test_labels, y_pred, args.output)

        print("\n分类报告:")
        print(classification_report(
            test_labels, 
            y_pred,
            target_names=[INVERSE_GESTURE_MAPPING[i] for i in range(len(GESTURE_MAPPING))]
        ))

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join(args.output, f"head_gesture_model_{timestamp}.h5")
        model.save(model_path)
        print(f"\n模型已保存到 {model_path}")

        scaler_path = os.path.join(args.output, f"scaler_{timestamp}.pkl")
        with open(scaler_path, 'wb') as f:
            pickle.dump(data_processor.scaler, f)
        print(f"标准化器已保存到 {scaler_path}")

        config = {
            "window_size": args.window,
            "stride": args.stride,
            "feature_dim": data_processor.feature_dim,
            "num_classes": len(GESTURE_MAPPING),
            "gesture_mapping": GESTURE_MAPPING,
            "accuracy": float(test_accuracy),
            "timestamp": timestamp
        }

        config_path = os.path.join(args.output, f"model_config_{timestamp}.json")
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=4, ensure_ascii=False)
        print(f"模型配置已保存到 {config_path}")

    except Exception as e:
        print(f"训练过程中出错: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
