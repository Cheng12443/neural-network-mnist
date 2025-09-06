import NeuralNetwork

if __name__ == '__main__':
    input_nodes = 3  # 输入层节点
    hidden_nodes = 15  # 隐藏层节点
    output_nodes = 20  # 输出层节点
    learning_rate = 0.01  # 学习率
    layer_sizes = [hidden_nodes]
    n = NeuralNetwork.Neural_Network(input_nodes, layer_sizes, output_nodes, learning_rate)  # 创建神经网络
    print(n.query([1.8, 9.5, 1.7]))
