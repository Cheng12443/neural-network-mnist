import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from matplotlib.font_manager import FontProperties

# 设置中文字体
try:
    # 对于 MacOS
    font = FontProperties(fname='/System/Library/Fonts/PingFang.ttc')
except:
    try:
        # 对于 Windows
        font = FontProperties(fname='C:\\Windows\\Fonts\\simhei.ttf')
    except:
        # 如果都找不到，使用系统默认字体
        font = FontProperties()

def load_data(filename):
    # 加载数据
    data = pd.read_csv(filename, header=None)
    # 分离特征和标签
    X = data.iloc[:, 1:].values / 255.0  # 归一化像素值
    y = data.iloc[:, 0].values
    # 将标签转换为one-hot编码
    y_one_hot = np.zeros((len(y), 10))
    for i, label in enumerate(y):
        y_one_hot[i][label] = 1
    return X, y, y_one_hot

def display_digit(image, label=None):
    # 显示28x28的手写数字图像
    plt.figure(figsize=(4,4))
    plt.imshow(image.reshape(28,28), cmap='gray')
    if label is not None:
        # 使用英文替代中文，或者使用设置的中文字体
        plt.title(f'Prediction: {label}', fontproperties=font)
    plt.axis('off')
    plt.show()

def main():
    # 加载测试数据
    print("加载测试数据...")
    X_test, y_test, y_test_one_hot = load_data('MNIST/mnist_test.csv')
    
    # 加载训练好的模型
    print("加载模型...")
    try:
        model = joblib.load('/Users/chengxingbang/Desktop/搭建人工神经网络/python/10000_trained_neural_network.pkl')
    except Exception as e:
        print(f"错误: 加载模型文件失败: {str(e)}")
        return
        
    # 进行预测
    print("开始预测...")
    predictions_prob = []
    batch_size = 100  # 设置批处理大小以加快处理速度
    
    for i in range(0, len(X_test), batch_size):
        batch = X_test[i:i+batch_size]
        batch_predictions = []
        for sample in batch:
            result = model.query(sample)
            batch_predictions.append(result.flatten())
        predictions_prob.extend(batch_predictions)
    
    predictions_prob = np.array(predictions_prob)
    predictions = np.argmax(predictions_prob, axis=1)
    
    
    # 随机显示几个预测结果
    print("\n显示一些预测结果示例:")
    n_samples = 20
    sample_indices = np.random.randint(0, len(X_test), n_samples)
    
    for idx in sample_indices:
        pred = predictions[idx]
        true_label = y_test[idx]
        print(f"\n预测值: {pred}")
        print(f"真实值: {true_label}")
        # display_digit(X_test[idx], label=f"Pred:{pred}, True:{true_label}")
    
        # 计算准确率
    accuracy = np.mean(predictions == y_test)
    print(f"测试集准确率: {accuracy*100:.2f}%")

if __name__ == "__main__":
    main() 