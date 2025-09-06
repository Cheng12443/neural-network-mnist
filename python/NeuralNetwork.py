import numpy
import scipy
import matplotlib.pyplot
from numba import jit  # 添加Numba支持


class Neural_Network():
    def __init__(self, inputnodes, hiddennodes_list, outputnodes, learningrate):
        """
        初始化神经网络
        inputnodes: 输入层节点数
        hiddennodes_list: 隐藏层节点数列表，例如[128, 64]表示两个隐藏层
        outputnodes: 输出层节点数
        learningrate: 学习率
        """
        # 初始化方法
        self.inodes = inputnodes
        self.hnodes_list = hiddennodes_list
        self.onodes = outputnodes
        self.lr = learningrate

        # 初始化权重
        self.weights = []
        
        # 输入层到第一个隐藏层的权重
        w_input = numpy.random.normal(0.0, pow(self.hnodes_list[0], -0.5), 
                                 (self.hnodes_list[0], self.inodes))
        self.weights.append(w_input)
        
        # 隐藏层之间的权重
        for i in range(len(self.hnodes_list)-1):
            w_hidden = numpy.random.normal(0.0, pow(self.hnodes_list[i+1], -0.5),
                                      (self.hnodes_list[i+1], self.hnodes_list[i]))
            self.weights.append(w_hidden)
        
        # 最后一个隐藏层到输出层的权重
        w_output = numpy.random.normal(0.0, pow(self.onodes, -0.5),
                                  (self.onodes, self.hnodes_list[-1]))
        self.weights.append(w_output)

        # 使用更快的激活函数
        self.activation_function = self.fast_sigmoid

    @staticmethod
    @jit(nopython=True)
    def fast_sigmoid(x):
        """使用Numba加速的sigmoid函数"""
        return 1.0 / (1.0 + numpy.exp(-numpy.clip(x, -500, 500)))
    
    def train(self, input_batch, target_batch):
        """训练网络"""
        inputs = numpy.array(input_batch, ndmin=2).T
        targets = numpy.array(target_batch, ndmin=2).T
        
        # 前向传播
        layer_outputs = [inputs]
        current_inputs = inputs
        
        for w in self.weights:
            hidden_inputs = numpy.clip(numpy.dot(w, current_inputs), -500, 500)
            hidden_outputs = self.activation_function(hidden_inputs)
            layer_outputs.append(hidden_outputs)
            current_inputs = hidden_outputs

        # 反向传播
        output_errors = targets - layer_outputs[-1]
        current_errors = output_errors

        # 更新权重
        weight_updates = []
        for i in range(len(self.weights)-1, -1, -1):
            gradient = numpy.clip(
                numpy.dot(
                    (current_errors * layer_outputs[i+1] * (1-layer_outputs[i+1])),
                    layer_outputs[i].T
                ),
                -1, 1
            )
            weight_updates.insert(0, self.weights[i] + self.lr * gradient)
            
            if i > 0:
                current_errors = numpy.dot(self.weights[i].T, current_errors)
        
        return weight_updates

    def query(self, input_list):
        """查询网络"""
        inputs = numpy.array(input_list, ndmin=2).T
        current_inputs = inputs
        
        for w in self.weights:
            layer_inputs = numpy.clip(numpy.dot(w, current_inputs), -500, 500)
            current_inputs = self.activation_function(layer_inputs)
        
        return current_inputs


# 示例使用
if __name__ == "__main__":
    # 创建一个具有两个隐藏层的网络
    input_nodes = 784
    hidden_nodes_list = [128, 64]  # 两个隐藏层，分别有128和64个节点
    output_nodes = 10
    learning_rate = 0.1
    
    # 初始化网络
    nn = Neural_Network(input_nodes, hidden_nodes_list, output_nodes, learning_rate)
    
    # 训练示例
    # input_data = ...  # 您的输入数据
    # target_data = ... # 对应的目标数据
    # nn.train(input_data, target_data)
    
    # 预测示例
    # test_data = ...   # 测试数据
    # result = nn.query(test_data)
