#include "neural_network.h"

// 定义全局学习率变量
double learning_rate = 0.1;

// sigmoid激活函数
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

// 初始化网络
NeuralNetwork* init_network() {
    NeuralNetwork* nn = (NeuralNetwork*)malloc(sizeof(NeuralNetwork));
    
    // 分配并初始化权重矩阵
    nn->w1 = (double**)malloc(HIDDEN_NODES_1 * sizeof(double*));
    nn->w2 = (double**)malloc(HIDDEN_NODES_2 * sizeof(double*));
    nn->w3 = (double**)malloc(OUTPUT_NODES * sizeof(double*));
    
    // 初始化第一层权重
    for(int i = 0; i < HIDDEN_NODES_1; i++) {
        nn->w1[i] = (double*)malloc(INPUT_NODES * sizeof(double));
        for(int j = 0; j < INPUT_NODES; j++) {
            nn->w1[i][j] = (rand() / (double)RAND_MAX - 0.5) * pow(HIDDEN_NODES_1, -0.5);
        }
    }
    
    // 初始化第二层权重
    for(int i = 0; i < HIDDEN_NODES_2; i++) {
        nn->w2[i] = (double*)malloc(HIDDEN_NODES_1 * sizeof(double));
        for(int j = 0; j < HIDDEN_NODES_1; j++) {
            nn->w2[i][j] = (rand() / (double)RAND_MAX - 0.5) * pow(HIDDEN_NODES_2, -0.5);
        }
    }
    
    // 初始化输出层权重
    for(int i = 0; i < OUTPUT_NODES; i++) {
        nn->w3[i] = (double*)malloc(HIDDEN_NODES_2 * sizeof(double));
        for(int j = 0; j < HIDDEN_NODES_2; j++) {
            nn->w3[i][j] = (rand() / (double)RAND_MAX - 0.5) * pow(OUTPUT_NODES, -0.5);
        }
    }
    
    return nn;
}

// 前向传播
void forward(NeuralNetwork* nn, double* input, double* output) {
    // 第一隐藏层
    double* hidden1 = (double*)calloc(HIDDEN_NODES_1, sizeof(double));
    for(int i = 0; i < HIDDEN_NODES_1; i++) {
        for(int j = 0; j < INPUT_NODES; j++) {
            hidden1[i] += nn->w1[i][j] * input[j];
        }
        hidden1[i] = sigmoid(hidden1[i]);
    }
    
    // 第二隐藏层
    double* hidden2 = (double*)calloc(HIDDEN_NODES_2, sizeof(double));
    for(int i = 0; i < HIDDEN_NODES_2; i++) {
        for(int j = 0; j < HIDDEN_NODES_1; j++) {
            hidden2[i] += nn->w2[i][j] * hidden1[j];
        }
        hidden2[i] = sigmoid(hidden2[i]);
    }
    
    // 输出层
    for(int i = 0; i < OUTPUT_NODES; i++) {
        output[i] = 0;
        for(int j = 0; j < HIDDEN_NODES_2; j++) {
            output[i] += nn->w3[i][j] * hidden2[j];
        }
        output[i] = sigmoid(output[i]);
    }
    
    free(hidden1);
    free(hidden2);
}

// 训练网络
void train(NeuralNetwork* nn, double* input, double* target) {
    // 存储前向传播的中间结果
    double* hidden1 = (double*)calloc(HIDDEN_NODES_1, sizeof(double));
    double* hidden2 = (double*)calloc(HIDDEN_NODES_2, sizeof(double));
    double* output = (double*)calloc(OUTPUT_NODES, sizeof(double));
    
    // 前向传播
    // 第一隐藏层
    for(int i = 0; i < HIDDEN_NODES_1; i++) {
        for(int j = 0; j < INPUT_NODES; j++) {
            hidden1[i] += nn->w1[i][j] * input[j];
        }
        hidden1[i] = sigmoid(hidden1[i]);
    }
    
    // 第二隐藏层
    for(int i = 0; i < HIDDEN_NODES_2; i++) {
        for(int j = 0; j < HIDDEN_NODES_1; j++) {
            hidden2[i] += nn->w2[i][j] * hidden1[j];
        }
        hidden2[i] = sigmoid(hidden2[i]);
    }
    
    // 输出层
    for(int i = 0; i < OUTPUT_NODES; i++) {
        for(int j = 0; j < HIDDEN_NODES_2; j++) {
            output[i] += nn->w3[i][j] * hidden2[j];
        }
        output[i] = sigmoid(output[i]);
    }
    
    // 计算输出层的误差
    double* output_errors = (double*)calloc(OUTPUT_NODES, sizeof(double));
    for(int i = 0; i < OUTPUT_NODES; i++) {
        output_errors[i] = target[i] - output[i];
    }
    
    // 计算隐藏层2的误差
    double* hidden2_errors = (double*)calloc(HIDDEN_NODES_2, sizeof(double));
    for(int i = 0; i < HIDDEN_NODES_2; i++) {
        double error = 0;
        for(int j = 0; j < OUTPUT_NODES; j++) {
            error += output_errors[j] * nn->w3[j][i];
        }
        hidden2_errors[i] = error;
    }
    
    // 计算隐藏层1的误差
    double* hidden1_errors = (double*)calloc(HIDDEN_NODES_1, sizeof(double));
    for(int i = 0; i < HIDDEN_NODES_1; i++) {
        double error = 0;
        for(int j = 0; j < HIDDEN_NODES_2; j++) {
            error += hidden2_errors[j] * nn->w2[j][i];
        }
        hidden1_errors[i] = error;
    }
    
    // 更新权重
    // 输出层到隐藏层2
    for(int i = 0; i < OUTPUT_NODES; i++) {
        for(int j = 0; j < HIDDEN_NODES_2; j++) {
            nn->w3[i][j] += learning_rate * output_errors[i] * output[i] * (1 - output[i]) * hidden2[j];
        }
    }
    
    // 隐藏层2到隐藏层1
    for(int i = 0; i < HIDDEN_NODES_2; i++) {
        for(int j = 0; j < HIDDEN_NODES_1; j++) {
            nn->w2[i][j] += learning_rate * hidden2_errors[i] * hidden2[i] * (1 - hidden2[i]) * hidden1[j];
        }
    }
    
    // 隐藏层1到输入层
    for(int i = 0; i < HIDDEN_NODES_1; i++) {
        for(int j = 0; j < INPUT_NODES; j++) {
            nn->w1[i][j] += learning_rate * hidden1_errors[i] * hidden1[i] * (1 - hidden1[i]) * input[j];
        }
    }
    
    // 释放内存
    free(hidden1);
    free(hidden2);
    free(output);
    free(output_errors);
    free(hidden2_errors);
    free(hidden1_errors);
}

// 保存模型
void save_model(NeuralNetwork* nn, const char* filename) {
    FILE* file = fopen(filename, "wb");
    if(file == NULL) {
        printf("无法创建文件\n");
        return;
    }
    
    // 保存权重
    for(int i = 0; i < HIDDEN_NODES_1; i++) {
        fwrite(nn->w1[i], sizeof(double), INPUT_NODES, file);
    }
    for(int i = 0; i < HIDDEN_NODES_2; i++) {
        fwrite(nn->w2[i], sizeof(double), HIDDEN_NODES_1, file);
    }
    for(int i = 0; i < OUTPUT_NODES; i++) {
        fwrite(nn->w3[i], sizeof(double), HIDDEN_NODES_2, file);
    }
    
    fclose(file);
}

// 加载模型
void load_model(NeuralNetwork* nn, const char* filename) {
    FILE* file = fopen(filename, "rb");
    if(file == NULL) {
        printf("无法打开文件\n");
        return;
    }
    
    // 读取权重
    for(int i = 0; i < HIDDEN_NODES_1; i++) {
        fread(nn->w1[i], sizeof(double), INPUT_NODES, file);
    }
    for(int i = 0; i < HIDDEN_NODES_2; i++) {
        fread(nn->w2[i], sizeof(double), HIDDEN_NODES_1, file);
    }
    for(int i = 0; i < OUTPUT_NODES; i++) {
        fread(nn->w3[i], sizeof(double), HIDDEN_NODES_2, file);
    }
    
    fclose(file);
}

// 释放网络内存
void free_network(NeuralNetwork* nn) {
    for(int i = 0; i < HIDDEN_NODES_1; i++) {
        free(nn->w1[i]);
    }
    for(int i = 0; i < HIDDEN_NODES_2; i++) {
        free(nn->w2[i]);
    }
    for(int i = 0; i < OUTPUT_NODES; i++) {
        free(nn->w3[i]);
    }
    
    free(nn->w1);
    free(nn->w2);
    free(nn->w3);
    free(nn);
} 