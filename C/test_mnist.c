#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define INPUT_NODES 784
#define OUTPUT_NODES 10
#define MAX_LINE_LENGTH 10000
#define TEST_SAMPLES 10000

// 加载CSV文件中的测试数据
void load_test_data(const char* filename, double** test_data, int* labels, int* num_samples) {
    FILE* file = fopen(filename, "r");
    if (file == NULL) {
        printf("无法打开文件: %s\n", filename);
        exit(1);
    }

    char line[MAX_LINE_LENGTH];
    int sample_index = 0;
    
    // 读取每一行数据
    while (fgets(line, MAX_LINE_LENGTH, file) && sample_index < TEST_SAMPLES) {
        char* token = strtok(line, ",");
        labels[sample_index] = atoi(token);  // 第一个值是标签
        
        // 读取图像像素值并归一化
        for (int i = 0; i < INPUT_NODES; i++) {
            token = strtok(NULL, ",");
            if (token == NULL) break;
            test_data[sample_index][i] = atof(token) / 255.0;
        }
        
        sample_index++;
    }
    
    *num_samples = sample_index;
    fclose(file);
}

// 从文件加载训练好的模型参数
void load_model(const char* filename, double** weights, double* biases) {
    FILE* file = fopen(filename, "rb");
    if (file == NULL) {
        printf("无法打开模型文件: %s\n", filename);
        exit(1);
    }
    
    // 读取权重和偏置
    for (int i = 0; i < OUTPUT_NODES; i++) {
        fread(weights[i], sizeof(double), INPUT_NODES, file);
    }
    fread(biases, sizeof(double), OUTPUT_NODES, file);
    
    fclose(file);
}

// sigmoid激活函数
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

// 预测单个样本
int predict(double* input, double** weights, double* biases) {
    double outputs[OUTPUT_NODES] = {0};
    
    // 计算输出层的值
    for (int i = 0; i < OUTPUT_NODES; i++) {
        double sum = biases[i];
        for (int j = 0; j < INPUT_NODES; j++) {
            sum += weights[i][j] * input[j];
        }
        outputs[i] = sigmoid(sum);
    }
    
    // 找出最大概率的类别
    int max_index = 0;
    double max_value = outputs[0];
    for (int i = 1; i < OUTPUT_NODES; i++) {
        if (outputs[i] > max_value) {
            max_value = outputs[i];
            max_index = i;
        }
    }
    
    return max_index;
}

int main() {
    // 使用动态内存分配
    double** test_data = (double**)malloc(TEST_SAMPLES * sizeof(double*));
    for (int i = 0; i < TEST_SAMPLES; i++) {
        test_data[i] = (double*)malloc(INPUT_NODES * sizeof(double));
    }
    
    int* true_labels = (int*)malloc(TEST_SAMPLES * sizeof(int));
    
    double** weights = (double**)malloc(OUTPUT_NODES * sizeof(double*));
    for (int i = 0; i < OUTPUT_NODES; i++) {
        weights[i] = (double*)malloc(INPUT_NODES * sizeof(double));
    }
    
    double* biases = (double*)malloc(OUTPUT_NODES * sizeof(double));
    
    int num_samples;
    
    // 初始化随机数生成器
    srand(time(NULL));
    
    printf("加载测试数据...\n");
    load_test_data("MNIST/mnist_test.csv", test_data, true_labels, &num_samples);
    
    printf("加载模型...\n");
    load_model("/Users/chengxingbang/Desktop/搭建人工神经网络/1000_trained_neural_network.pkl", weights, biases);
    
    printf("开始预测...\n");
    int correct = 0;
    
    // 测试20个随机样本
    printf("\n显示20个预测结果示例:\n");
    for (int i = 0; i < 20; i++) {
        int idx = rand() % num_samples;
        int prediction = predict(test_data[idx], weights, biases);
        printf("\n预测值: %d", prediction);
        printf("\n真实值: %d\n", true_labels[idx]);
    }
    
    // 计算整体准确率
    for (int i = 0; i < num_samples; i++) {
        int prediction = predict(test_data[i], weights, biases);
        if (prediction == true_labels[i]) {
            correct++;
        }
    }
    
    double accuracy = (double)correct / num_samples * 100;
    printf("\n测试集准确率: %.2f%%\n", accuracy);
    
    // 释放内存
    for (int i = 0; i < TEST_SAMPLES; i++) {
        free(test_data[i]);
    }
    free(test_data);
    free(true_labels);
    
    for (int i = 0; i < OUTPUT_NODES; i++) {
        free(weights[i]);
    }
    free(weights);
    free(biases);
    
    return 0;
} 