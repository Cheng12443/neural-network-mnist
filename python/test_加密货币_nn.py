from tqdm import tqdm
import numpy as np
import pickle5
import json
import csv
from NeuralNetwork import Neural_Network


def load_crypto_data(file_path):
    """加载加密货币数据"""
    X, y = [], []
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # 跳过表头
        for row in reader:
            X.append(json.loads(row[0]))
            y.append(json.loads(row[1]))
    return np.array(X), np.array(y)


def main():
    # 加载数据
    print("加载数据...")
    X, y = load_crypto_data('/Users/chengxingbang/Desktop/搭建人工神经网络/python/pseudo_data.csv')

    # 数据集大小
    print(f"\n数据集大小:")
    print(f"输入数据形状: {X.shape}")
    print(f"输出数据形状: {y.shape}")

    # 划分训练集和测试集
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # 创建神经网络
    input_nodes = X.shape[1]  # 输入维度（5个历史价格变化）
    hidden_nodes_list = [32, 16]  # 两个隐藏层
    output_nodes = 1  # 输出维度（下一个价格变化）
    learning_rate = 0.01

    print("\n创建神经网络...")
    print(f"输入节点: {input_nodes}")
    print(f"隐藏层: {hidden_nodes_list}")
    print(f"输出节点: {output_nodes}")
    print(f"学习率: {learning_rate}")

    nn = Neural_Network(input_nodes, hidden_nodes_list, output_nodes, learning_rate)

    # 训练网络
    print("\n开始训练...")
    epochs = 100
    batch_size = 32

    for epoch in tqdm(range(epochs)):
        total_loss = 0

        # 小批量训练
        for i in range(0, len(X_train), batch_size):
            batch_X = X_train[i:i + batch_size]
            batch_y = y_train[i:i + batch_size]

            for j in range(len(batch_X)):
                # 训练单个样本
                weights = nn.train(batch_X[j], batch_y[j])
                if weights:  # 如果有返回的权重更新
                    nn.weights = weights

            # 计算这个批次的损失
            batch_predictions = np.array([nn.query(x).flatten() for x in batch_X])
            batch_loss = np.mean((batch_predictions - batch_y.flatten()) ** 2)
            total_loss += batch_loss

        # 每100轮打印一次损失
        if (epoch + 1) % 100 == 0:
            avg_loss = total_loss / (len(X_train) / batch_size)
            print(f"\n轮次 {epoch + 1}/{epochs}, 平均损失: {avg_loss:.6f}")

    print("\nSaving model...")
    with open(f'{epochs}_Nodle_trained_neural_network.pkl', 'wb') as f:
        pickle5.dump(nn, f)

    # 测试网络
    print("\n测试网络...")
    test_predictions = np.array([nn.query(x).flatten()[0] for x in X_test])
    test_loss = np.mean((test_predictions - y_test.flatten()) ** 2)
    print(f"测试集损失: {test_loss:.6f}")

    # 预测示例
    print("\n预测示例:")
    for i in range(10):  # 显示前5个预测
        print(f"\n输入序列: {X_test[i].tolist()}")
        print(f"实际变化: {y_test[i][0]:.6f}")
        print(f"预测变化: {test_predictions[i]:.6f}")
        print(f"预测误差: {abs(y_test[i][0] - test_predictions[i]):.6f}")


if __name__ == "__main__":
    main()
