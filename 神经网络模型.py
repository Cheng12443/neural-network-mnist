import numpy as np
import json
import csv


# 神经网络类（简化版，见之前代码）
def sigmoid_derivative(x):
    return x * (1 - x)


def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


# 添加新的数据预处理函数
def normalize_data(X, y):
    # 计算均值和标准差
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    y_mean = np.mean(y, axis=0)
    y_std = np.std(y, axis=0)

    # 标准化数据
    X_normalized = (X - X_mean) / (X_std + 1e-8)
    y_normalized = (y - y_mean) / (y_std + 1e-8)

    return X_normalized, y_normalized, (X_mean, X_std), (y_mean, y_std)


class SimpleNN:
    def __init__(self, input_size, output_size, hidden_size=64, learning_rate=0.01):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate

        # 使用Xavier初始化
        self.weights_input_hidden = np.random.randn(input_size, hidden_size) / np.sqrt(input_size)
        self.bias_hidden = np.zeros(hidden_size)
        self.weights_hidden_output = np.random.randn(hidden_size, output_size) / np.sqrt(hidden_size)
        self.bias_output = np.zeros(output_size)

    def forward(self, x):
        # 前向传播
        self.input = x
        self.hidden = sigmoid(np.dot(x, self.weights_input_hidden) + self.bias_hidden)
        self.output = np.dot(self.hidden, self.weights_hidden_output) + self.bias_output
        return self.output

    def backward(self, target):
        # 计算损失的梯度
        output_error = target - self.output
        output_delta = output_error

        hidden_error = np.dot(output_delta, self.weights_hidden_output.T)
        hidden_delta = hidden_error * sigmoid_derivative(self.hidden)

        # 添加梯度裁剪
        clip_value = 5.0
        output_delta = np.clip(output_delta, -clip_value, clip_value)
        hidden_delta = np.clip(hidden_delta, -clip_value, clip_value)

        # 更新权重和偏置
        self.weights_hidden_output += self.learning_rate * np.dot(self.hidden.T, output_delta)
        self.bias_output += self.learning_rate * np.sum(output_delta, axis=0)

        self.weights_input_hidden += self.learning_rate * np.dot(self.input.T, hidden_delta)
        self.bias_hidden += self.learning_rate * np.sum(hidden_delta, axis=0)

    def train(self, X, y, epochs=1000, batch_size=32):
        n_samples = X.shape[0]
        n_batches = n_samples // batch_size

        for epoch in range(epochs):
            total_loss = 0
            # 随机打乱数据
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            for batch in range(n_batches):
                start_idx = batch * batch_size
                end_idx = start_idx + batch_size

                X_batch = X_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]

                output = self.forward(X_batch)
                self.backward(y_batch)

                batch_loss = np.mean((y_batch - output) ** 2)
                total_loss += batch_loss

            if (epoch + 1) % 100 == 0:
                avg_loss = total_loss / n_batches
                print(f"第 {epoch + 1} 轮训练, 损失: {avg_loss:.4f}")

    def save_model(self, file_path, data_stats=None):
        # 保存模型权重、偏置和数据统计信息到文件
        model_data = {
            "weights_input_hidden": self.weights_input_hidden.tolist(),
            "bias_hidden": self.bias_hidden.tolist(),
            "weights_hidden_output": self.weights_hidden_output.tolist(),
            "bias_output": self.bias_output.tolist(),
            "input_size": self.input_size,
            "output_size": self.output_size,
            "hidden_size": self.hidden_size,
            "learning_rate": self.learning_rate
        }

        # 添加数据标准化参数
        if data_stats:
            model_data["data_stats"] = {
                "X_mean": data_stats[0][0].tolist(),
                "X_std": data_stats[0][1].tolist(),
                "y_mean": data_stats[1][0].tolist(),
                "y_std": data_stats[1][1].tolist()
            }

        with open(file_path, "w") as file:
            json.dump(model_data, file, indent=4)
        print(f"模型保存到: {file_path}")

    def load_model(self, file_path):
        # 从文件加载模型权重、偏置和数据统计信息
        with open(file_path, "r") as file:
            model_data = json.load(file)

        self.weights_input_hidden = np.array(model_data["weights_input_hidden"])
        self.bias_hidden = np.array(model_data["bias_hidden"])
        self.weights_hidden_output = np.array(model_data["weights_hidden_output"])
        self.bias_output = np.array(model_data["bias_output"])
        self.input_size = model_data["input_size"]
        self.output_size = model_data["output_size"]
        self.hidden_size = model_data["hidden_size"]
        self.learning_rate = model_data["learning_rate"]

        data_stats = None
        if "data_stats" in model_data:
            data_stats = (
                (np.array(model_data["data_stats"]["X_mean"]),
                 np.array(model_data["data_stats"]["X_std"])),
                (np.array(model_data["data_stats"]["y_mean"]),
                 np.array(model_data["data_stats"]["y_std"]))
            )

        print(f"模型加载自: {file_path}")
        return data_stats


# 随机生成伪数据
def generate_pseudo_data(input_dim, output_dim, num_samples, output_file):
    data = []
    current_input = np.random.rand(input_dim)  # 初始化随机输入
    for _ in range(num_samples):
        next_output = current_input + np.random.uniform(-0.1, 0.1, output_dim)  # 添加微小扰动
        data.append((current_input.tolist(), next_output.tolist()))
        current_input = next_output  # 下一次输入等于当前输出

    # 保存数据到 CSV 文件
    with open(output_file, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Input"] + ["Output"])  # 写入表头
        for inputs, outputs in data:
            writer.writerow([json.dumps(inputs), json.dumps(outputs)])

    print(f"Pseudo data saved to {output_file}")
    print(data)
    return data


# 加载 CSV 数据用于训练
def load_data_from_csv(file_path):
    X, y = [], []
    with open(file_path, "r") as file:
        reader = csv.reader(file)
        next(reader)  # 跳过表头
        for row in reader:
            X.append(json.loads(row[0]))
            y.append(json.loads(row[1]))
    return np.array(X), np.array(y)


# 示例使用
if __name__ == "__main__":
    input_dim = 5
    output_dim = 5
    num_samples = 10000
    output_file = "pseudo_data.csv"

    # 生成伪数据
    generate_pseudo_data(input_dim, output_dim, num_samples, output_file)

    # 加载并预处理数据
    X, y = load_data_from_csv(output_file)
    X_norm, y_norm, X_stats, y_stats = normalize_data(X, y)

    # 初始化模型并训练
    model = SimpleNN(
        input_dim,
        output_dim,
        hidden_size=64,  # 增加隐藏层大小
        learning_rate=0.05  # 调整学习率
    )

    # 使用小批量训练
    model.train(X_norm, y_norm, epochs=10000, batch_size=32)

    # 训练完成后保存模型
    model_file = "neural_network_model.json"
    data_stats = (X_stats, y_stats)
    model.save_model(model_file, data_stats)

    # 加载模型示例
    new_model = SimpleNN(input_dim, output_dim)
    loaded_stats = new_model.load_model(model_file)

    # 使用加载的模型进行预测
    if loaded_stats:
        X_stats, y_stats = loaded_stats
        test_input = np.random.rand(input_dim)
        # 使用加载的标准化参数
        test_input_norm = (test_input - X_stats[0]) / (X_stats[1] + 1e-8)
        predicted_output_norm = new_model.forward(test_input_norm)
        predicted_output = predicted_output_norm * (y_stats[1] + 1e-8) + y_stats[0]

        print("\n使用加载的模型进行预测:")
        print("测试输入:", test_input)
        print("预测输出:", predicted_output)
