import dill as pickle5
import numpy
from tqdm import tqdm
from NeuralNetwork import Neural_Network
import multiprocessing as mp
from functools import partial


def train_batch(network, batch_data):
    """训练单个批次"""
    batch_inputs, batch_targets = batch_data
    return network.train(batch_inputs, batch_targets)


def main():
    # 网络参数
    input_nodes = 784
    hidden_nodes_list = [200, 100]
    output_nodes = 10
    learning_rate = 0.01
    batch_size = 128
    num_processes = mp.cpu_count()

    # 创建神经网络
    neural_network = Neural_Network(input_nodes, hidden_nodes_list, output_nodes, learning_rate)

    # 加载数据
    print("Loading and preprocessing data...")
    training_data = numpy.loadtxt('/Users/chengxingbang/Desktop/搭建人工神经网络/MNIST/mnist_train.csv',
                                  delimiter=',')

    # 预处理数据
    training_inputs = (training_data[:, 1:] / 255.0 * 0.99) + 0.01
    training_targets = numpy.zeros((training_data.shape[0], output_nodes)) + 0.01
    training_targets[numpy.arange(training_data.shape[0]),
    training_data[:, 0].astype(int)] = 0.99

    epochs = 100000
    n_samples = len(training_inputs)

    # 创建进程池
    pool = mp.Pool(processes=num_processes)

    print("Starting training...")
    for e in tqdm(range(epochs)):
        # 随机打乱数据
        indices = numpy.random.permutation(n_samples)
        training_inputs = training_inputs[indices]
        training_targets = training_targets[indices]

        # 准备批次数据
        batches = [(training_inputs[i:i + batch_size],
                    training_targets[i:i + batch_size])
                   for i in range(0, n_samples, batch_size)]

        # 并行处理批次
        train_func = partial(train_batch, neural_network)
        weights_updates = pool.map(train_func, batches)

        # 更新权重（取平均）
        for i in range(len(neural_network.weights)):
            neural_network.weights[i] = numpy.mean([w[i] for w in weights_updates], axis=0)

        # 计算准确率
        if e % 30 == 0:
            predictions = neural_network.query(training_inputs[:1000])
            accuracy = numpy.mean(numpy.argmax(predictions, axis=0) ==
                                  training_data[:1000, 0])
            print(f"\nEpoch {e}, Accuracy: {accuracy:.4f}")

    pool.close()
    pool.join()

    # 保存模型
    print("Saving model...")
    with open(f'{epochs}_trained_neural_network.pkl', 'wb') as f:
        pickle5.dump(neural_network, f)


if __name__ == '__main__':
    main()
