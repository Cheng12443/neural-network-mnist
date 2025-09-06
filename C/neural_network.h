#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// 定义网络结构
#define INPUT_NODES 784
#define HIDDEN_NODES_1 128
#define HIDDEN_NODES_2 64
#define OUTPUT_NODES 10

// 将学习率改为外部全局变量
extern double learning_rate;

typedef struct {
    // 权重矩阵
    double** w1;  // 输入层到第一隐藏层
    double** w2;  // 第一隐藏层到第二隐藏层
    double** w3;  // 第二隐藏层到输出层
} NeuralNetwork;

// 函数声明
NeuralNetwork* init_network();
void forward(NeuralNetwork* nn, double* input, double* output);
void train(NeuralNetwork* nn, double* input, double* target);
void save_model(NeuralNetwork* nn, const char* filename);
void load_model(NeuralNetwork* nn, const char* filename);
void free_network(NeuralNetwork* nn);

#endif 