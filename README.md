# 人工神经网络项目

这是一个包含C语言和Python实现的人工神经网络项目，支持MNIST数据集训练和预测，以及加密货币价格预测。

## 项目结构

```
搭建人工神经网络/
├── C/                          # C语言实现
│   ├── neural_network.c       # 核心神经网络实现
│   ├── neural_network.h       # 头文件
│   ├── train_mnist.c          # MNIST训练程序
│   └── test_mnist.c           # MNIST测试程序
├── python/                     # Python实现
│   ├── NeuralNetwork.py       # 神经网络类
│   ├── train_mnist.py         # MNIST训练脚本
│   ├── test_mnist.py          # MNIST测试脚本
│   ├── train_nodl.py          # 加密货币训练
│   └── predict_nodl.py        # 加密货币预测
├── MNIST/                      # MNIST数据集
├── models/                     # 训练好的模型
└── README.md                   # 项目说明
```

## 功能特性

- **C语言实现**: 高效的C语言神经网络实现
- **Python实现**: 基于Keras/TensorFlow的高级API
- **MNIST支持**: 完整的手写数字识别支持
- **加密货币预测**: 支持加密货币价格趋势预测
- **多语言支持**: C和Python两种实现方式

## 使用方法

### C语言版本

```bash
cd C
make
./train_mnist
./test_mnist
```

### Python版本

```bash
cd python
python train_mnist.py
python test_mnist.py
```

### 加密货币预测

```bash
cd python
python train_nodl.py
python predict_nodl.py
```

## 依赖项

### C语言依赖
- GCC编译器
- Make
- 标准C库

### Python依赖
- Python 3.7+
- TensorFlow 2.x
- Keras
- NumPy
- Pandas
- Matplotlib

## 安装

1. 克隆仓库：
```bash
git clone https://github.com/Cheng12443/neural-network-mnist.git
```

2. 安装Python依赖：
```bash
pip install -r requirements.txt
```

3. 运行示例：
```bash
cd neural-network-mnist
python python/train_mnist.py
```

## 数据集

- **MNIST**: 手写数字识别数据集，包含60,000个训练样本和10,000个测试样本
- **加密货币数据**: 历史价格数据用于训练和预测

### 数据集下载

由于GitHub文件大小限制，MNIST数据集需要单独下载：

1. **下载MNIST数据集**：
```bash
cd MNIST
wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
gunzip *.gz
```

2. **或者使用Python自动下载**：
```bash
python python/train_mnist.py --download
```

## 模型性能

- MNIST测试准确率: >98%
- 加密货币预测准确率: 根据市场条件变化

## 贡献

欢迎提交Issue和Pull Request来改进这个项目。

## 许可证

MIT License