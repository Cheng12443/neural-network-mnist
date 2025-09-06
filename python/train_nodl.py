import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, utils
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pickle
import os
import json

# 数据预处理
def load_mnist_data(data_path):
    print(f"正在读取MNIST数据: {data_path}")
    df = pd.read_csv(data_path)
    y = df.iloc[:, 0].values
    X = df.iloc[:, 1:].values.reshape(-1, 28, 28, 1)
    return X, y

# 构建CNN模型
def build_cnn_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        layers.Dense(num_classes, activation='softmax')
    ])
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, utils
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pickle
import os
import json

# 数据预处理
def load_mnist_data(data_path):
    print(f"正在读取MNIST数据: {data_path}")
    df = pd.read_csv(data_path)
    y = df.iloc[:, 0].values
    X = df.iloc[:, 1:].values.reshape(-1, 28, 28, 1)
    return X, y

# 构建CNN模型
def build_cnn_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        layers.Dense(num_classes, activation='softmax')
    ])

    # 数据增强层
    data_augmentation = tf.keras.Sequential([
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1)
    ])

    # 使用正式版AdamW优化器
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=0.001,
        weight_decay=1e-4
    )
    
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'],
                  jit_compile=True)  # 启用XLA编译加速
    
    return model
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, utils
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pickle
import os
import json

# 数据预处理
def load_mnist_data(data_path):
    print(f"正在读取MNIST数据: {data_path}")
    df = pd.read_csv(data_path)
    y = df.iloc[:, 0].values
    X = df.iloc[:, 1:].values.reshape(-1, 28, 28, 1)
    return X, y

# 构建CNN模型
def build_cnn_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        layers.Dense(num_classes, activation='softmax')
    ])
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, utils
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pickle
import os
import json

# 数据预处理
def load_mnist_data(data_path):
    print(f"正在读取MNIST数据: {data_path}")
    df = pd.read_csv(data_path)
    y = df.iloc[:, 0].values
    X = df.iloc[:, 1:].values.reshape(-1, 28, 28, 1)
    return X, y

# 构建CNN模型（优化版）
def build_cnn_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        
        layers.Dense(num_classes, activation='softmax')
    ])

    # 使用带权重衰减的Adam优化器
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=0.001,
        weight_decay=1e-4
    )
    
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
        jit_compile=True  # 启用XLA加速
    )
    
    return model
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, utils
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pickle
import os
import json

# 数据预处理
def load_mnist_data(data_path):
    print(f"正在读取MNIST数据: {data_path}")
    df = pd.read_csv(data_path)
    y = df.iloc[:, 0].values
    X = df.iloc[:, 1:].values.reshape(-1, 28, 28, 1)
    return X, y

# 构建CNN模型
def build_cnn_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        layers.Dense(num_classes, activation='softmax')
    ])
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, utils
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pickle
import os
import json

# 数据预处理
def load_mnist_data(data_path):
    print(f"正在读取MNIST数据: {data_path}")
    df = pd.read_csv(data_path)
    y = df.iloc[:, 0].values
    X = df.iloc[:, 1:].values.reshape(-1, 28, 28, 1)
    return X, y

# 构建CNN模型
def build_cnn_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        layers.Dense(num_classes, activation='softmax')
    ])

    # 使用带权重衰减的Adam优化器
    optimizer = tf.optimizers.experimental.AdamW(learning_rate=0.001, weight_decay=1e-4)
    
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'],
                  jit_compile=True)  # 启用XLA编译加速
    
    return model
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, utils
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pickle
import os
import json

# 数据预处理
def load_mnist_data(data_path):
    print(f"正在读取MNIST数据: {data_path}")
    df = pd.read_csv(data_path)
    y = df.iloc[:, 0].values
    X = df.iloc[:, 1:].values.reshape(-1, 28, 28, 1)
    return X, y

# 构建CNN模型
def build_cnn_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        layers.Dense(num_classes, activation='softmax')
    ])
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, utils
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pickle
import os
import json

# 数据预处理
def load_mnist_data(data_path):
    print(f"正在读取MNIST数据: {data_path}")
    df = pd.read_csv(data_path)
    y = df.iloc[:, 0].values
    X = df.iloc[:, 1:].values.reshape(-1, 28, 28, 1)
    return X, y

# 构建CNN模型
def build_cnn_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

# 训练模型
def train_model(X_train, y_train, X_val, y_val):
    # 数据标准化
    X_train = X_train.astype('float32') / 255
    X_val = X_val.astype('float32') / 255

    # 创建优化后的回调
    callbacks = [
        callbacks.EarlyStopping(patience=15, 
                              monitor='val_accuracy',
                              mode='max',
                              restore_best_weights=True),
        callbacks.ModelCheckpoint('models/best_model.keras', 
                                save_best_only=True,
                                save_weights_only=False),
        callbacks.CSVLogger('logs/training_log.csv'),
        callbacks.ReduceLROnPlateau(monitor='val_loss', 
                                  factor=0.2,
                                  patience=5,
                                  min_lr=1e-6),
        callbacks.TensorBoard(log_dir='logs/tensorboard', 
                            histogram_freq=1)
    ]
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, utils
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pickle
import os
import json

# 数据预处理
def load_mnist_data(data_path):
    print(f"正在读取MNIST数据: {data_path}")
    df = pd.read_csv(data_path)
    y = df.iloc[:, 0].values
    X = df.iloc[:, 1:].values.reshape(-1, 28, 28, 1)
    return X, y

# 构建CNN模型
def build_cnn_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        layers.Dense(num_classes, activation='softmax')
    ])
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, utils
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pickle
import os
import json

# 数据预处理
def load_mnist_data(data_path):
    print(f"正在读取MNIST数据: {data_path}")
    df = pd.read_csv(data_path)
    y = df.iloc[:, 0].values
    X = df.iloc[:, 1:].values.reshape(-1, 28, 28, 1)
    return X, y

# 构建CNN模型
def build_cnn_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        layers.Dense(num_classes, activation='softmax')
    ])

    # 使用带权重衰减的Adam优化器
    optimizer = tf.optimizers.experimental.AdamW(learning_rate=0.001, weight_decay=1e-4)
    
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'],
                  jit_compile=True)  # 启用XLA编译加速
    
    return model
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, utils
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pickle
import os
import json

# 数据预处理
def load_mnist_data(data_path):
    print(f"正在读取MNIST数据: {data_path}")
    df = pd.read_csv(data_path)
    y = df.iloc[:, 0].values
    X = df.iloc[:, 1:].values.reshape(-1, 28, 28, 1)
    return X, y

# 构建CNN模型
def build_cnn_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        layers.Dense(num_classes, activation='softmax')
    ])
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, utils
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pickle
import os
import json

# 数据预处理
def load_mnist_data(data_path):
    print(f"正在读取MNIST数据: {data_path}")
    df = pd.read_csv(data_path)
    y = df.iloc[:, 0].values
    X = df.iloc[:, 1:].values.reshape(-1, 28, 28, 1)
    return X, y

# 构建CNN模型（优化版）
def build_cnn_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        
        layers.Dense(num_classes, activation='softmax')
    ])

    # 使用带权重衰减的Adam优化器
    optimizer = tf.keras.optimizers.AdamW(
        learning_rate=0.001,
        weight_decay=1e-4
    )
    
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
        jit_compile=True  # 启用XLA加速
    )
    
    return model
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, utils
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pickle
import os
import json

# 数据预处理
def load_mnist_data(data_path):
    print(f"正在读取MNIST数据: {data_path}")
    df = pd.read_csv(data_path)
    y = df.iloc[:, 0].values
    X = df.iloc[:, 1:].values.reshape(-1, 28, 28, 1)
    return X, y

# 构建CNN模型
def build_cnn_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        layers.Dense(num_classes, activation='softmax')
    ])
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, utils
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pickle
import os
import json

# 数据预处理
def load_mnist_data(data_path):
    print(f"正在读取MNIST数据: {data_path}")
    df = pd.read_csv(data_path)
    y = df.iloc[:, 0].values
    X = df.iloc[:, 1:].values.reshape(-1, 28, 28, 1)
    return X, y

# 构建CNN模型
def build_cnn_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        layers.Dense(num_classes, activation='softmax')
    ])

    # 使用带权重衰减的Adam优化器
    optimizer = tf.optimizers.experimental.AdamW(learning_rate=0.001, weight_decay=1e-4)
    
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'],
                  jit_compile=True)  # 启用XLA编译加速
    
    return model
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, utils
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pickle
import os
import json

# 数据预处理
def load_mnist_data(data_path):
    print(f"正在读取MNIST数据: {data_path}")
    df = pd.read_csv(data_path)
    y = df.iloc[:, 0].values
    X = df.iloc[:, 1:].values.reshape(-1, 28, 28, 1)
    return X, y

# 构建CNN模型
def build_cnn_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        layers.Dense(num_classes, activation='softmax')
    ])
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, utils
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pickle
import os
import json

# 数据预处理
def load_mnist_data(data_path):
    print(f"正在读取MNIST数据: {data_path}")
    df = pd.read_csv(data_path)
    y = df.iloc[:, 0].values
    X = df.iloc[:, 1:].values.reshape(-1, 28, 28, 1)
    return X, y

# 构建CNN模型
def build_cnn_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model

# 训练模型
def train_model(X_train, y_train, X_val, y_val):
    # 数据标准化
    X_train = X_train.astype('float32') / 255
    X_val = X_val.astype('float32') / 255

    # 创建训练目录
    os.makedirs('models', exist_ok=True)
    os.makedirs('logs/tensorboard', exist_ok=True)

    # 配置回调功能
    callback_list = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=15,
            mode='max',
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath='models/best_model.keras',
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=False,
            mode='max',
            verbose=1
        ),
        tf.keras.callbacks.CSVLogger(
            'logs/training_log.csv', 
            separator=',', 
            append=False
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        ),
        tf.keras.callbacks.TensorBoard(
            log_dir='logs/tensorboard',
            histogram_freq=1,
            update_freq='epoch'
        )
    ]

    # 创建模型
    model = build_cnn_model((28, 28, 1), 10)
    
    # 训练模型
    history = model.fit(X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=100,
                        batch_size=128,
                        callbacks=callbacks)
    
    # 保存训练曲线
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='训练准确率')
    plt.plot(history.history['val_accuracy'], label='验证准确率')
    plt.title('模型准确率')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='训练损失')
    plt.plot(history.history['val_loss'], label='验证损失')
    plt.title('模型损失')
    plt.legend()
    
    plt.savefig('training_metrics.png')
    plt.close()
    
    return model

def main():
    # 初始化日志
    from datetime import datetime
    start_time = datetime.now()
    
    try:
        # 加载数据
        X, y = load_mnist_data('MNIST/mnist_train.csv')
        
        # 分割数据集
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        print(f"训练集大小: {len(X_train)}")
        print(f"验证集大小: {len(X_val)}")

        # 训练模型
        model = train_model(X_train, y_train, X_val, y_val)
        
        # 保存最终模型
        version = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = f'models/mnist_cnn_{version}.keras'
        model.save(model_path)
        
        # 评估模型
        test_loss, test_acc = model.evaluate(X_val, y_val, verbose=0)
        print(f"\n测试准确率: {test_acc:.4f}")
        
        # 保存元数据
        metadata = {
            'version': version,
            'training_date': start_time.strftime("%Y-%m-%d %H:%M:%S"),
            'test_accuracy': test_acc,
            'input_shape': (28, 28, 1),
            'classes': 10
        }
        
        with open(f'models/metadata_{version}.json', 'w') as f:
            json.dump(metadata, f)
            
    except Exception as e:
        print(f"训练出错: {str(e)}")
        raise

if __name__ == "__main__":
    main()
