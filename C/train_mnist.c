#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "neural_network.h"

// 用于显示进度条的函数
void print_progress_bar(int epoch, int total_epochs, float loss, double elapsed_time) {
    const int bar_width = 50;
    float ratio = epoch / (float)total_epochs;
    int filled = (int)(bar_width * ratio);
    
    // 计算预计剩余时间
    double eta = (elapsed_time / epoch) * (total_epochs - epoch);
    int eta_h = (int)(eta / 3600);
    int eta_m = (int)((eta - eta_h * 3600) / 60);
    int eta_s = (int)(eta - eta_h * 3600 - eta_m * 60);
    
    printf("\r训练进度: [");
    for (int i = 0; i < bar_width; i++) {
        if (i < filled) {
            printf("=");
        } else {
            printf(" ");
        }
    }
    printf("] %3d%% | Loss: %.4f | 剩余时间: %02d:%02d:%02d", 
           (int)(ratio * 100), loss, eta_h, eta_m, eta_s);
    fflush(stdout);
}

int main() {
    srand(time(NULL));
    
    // 创建神经网络
    NeuralNetwork* nn = init_network();
    
    // 读取MNIST训练数据
    FILE* train_file = fopen("/Users/chengxingbang/Desktop/搭建人工神经网络/MNIST/mnist_train.csv", "r");
    if(train_file == NULL) {
        printf("无法打开训练数据文件\n");
        return 1;
    }
    
    // 训练参数
    int epochs = 10;
    double initial_learning_rate = learning_rate;
    char line[8192];
    
    printf("\n开始训练...\n");
    time_t start_time = time(NULL);
    
    for(int epoch = 0; epoch < epochs; epoch++) {
        // 学习率衰减
        learning_rate = initial_learning_rate / (1 + epoch * 0.1);
        
        // 训练阶段
        int train_total = 0;
        double total_loss = 0.0;
        rewind(train_file);
        
        while(fgets(line, sizeof(line), train_file)) {
            double input[INPUT_NODES] = {0};
            double target[OUTPUT_NODES] = {0.01};
            
            // 解析CSV行
            char* token = strtok(line, ",");
            int label = atoi(token);
            target[label] = 0.99;
            
            for(int i = 0; i < INPUT_NODES; i++) {
                token = strtok(NULL, ",");
                if(token == NULL) break;
                input[i] = (atof(token) / 255.0 * 0.99) + 0.01;
            }
            
            // 前向传播和训练
            double output[OUTPUT_NODES];
            forward(nn, input, output);
            
            // 计算损失
            double loss = 0.0;
            for(int i = 0; i < OUTPUT_NODES; i++) {
                loss += -target[i] * log(output[i] + 1e-10) - 
                        (1 - target[i]) * log(1 - output[i] + 1e-10);
            }
            total_loss += loss;
            
            train(nn, input, target);
            train_total++;
        }
        
        // 显示整体进度
        float avg_loss = total_loss / train_total;
        double elapsed_time = difftime(time(NULL), start_time);
        print_progress_bar(epoch + 1, epochs, avg_loss, elapsed_time);
    }
    
    // 计算总训练时间
    double total_time = difftime(time(NULL), start_time);
    int total_h = (int)(total_time / 3600);
    int total_m = (int)((total_time - total_h * 3600) / 60);
    int total_s = (int)(total_time - total_h * 3600 - total_m * 60);
    printf("\n\n总训练时间: %02d:%02d:%02d\n", total_h, total_m, total_s);
    
    // 计算最终准确率
    printf("计算训练集准确率...\n");
    int correct = 0;
    int total = 0;
    rewind(train_file);
    
    while(fgets(line, sizeof(line), train_file)) {
        double input[INPUT_NODES] = {0};
        
        // 解析CSV行
        char* token = strtok(line, ",");
        int label = atoi(token);
        
        for(int i = 0; i < INPUT_NODES; i++) {
            token = strtok(NULL, ",");
            if(token == NULL) break;
            input[i] = (atof(token) / 255.0 * 0.99) + 0.01;
        }
        
        // 前向传播
        double output[OUTPUT_NODES];
        forward(nn, input, output);
        
        // 计算准确率
        int predicted = 0;
        double max_val = output[0];
        for(int i = 1; i < OUTPUT_NODES; i++) {
            if(output[i] > max_val) {
                max_val = output[i];
                predicted = i;
            }
        }
        if(predicted == label) correct++;
        total++;
    }
    
    float final_accuracy = (float)correct / total * 100;
    printf("\n训练完成!\n");
    printf("最终训练准确率: %.2f%%\n", final_accuracy);
    
    // 保存最终模型
    save_model(nn, "trained_model.bin");
    printf("模型已保存为: trained_model.bin\n");
    
    // 清理
    fclose(train_file);
    free_network(nn);
    
    return 0;
} 