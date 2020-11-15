# -*- coding: utf-8 -*-
# File  : obtain_the_trained_network.py
# Author: Meijin Lu
# Date  : 2020/10/23
import neural_network as nn
import numpy

input_nodes = 784
hidden_nodes = 150
output_nodes = 10
learning_rate = 0.15
# 创建神经网络实例对象
my_nn = nn.neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)
# 从csv文件加载训练所需数据
training_data_file = open("data/mnist_train.csv", "r")
training_data_list = training_data_file.readlines()
training_data_file.close()
# 开始训练
epoches = 1  # 世代
for e in range(epoches):
    print("第{}世代开始训练".format(e+1))
    for record in training_data_list:
        all_values = record.split(",")
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        targets = (numpy.zeros(output_nodes) + 0.01)
        targets[int(all_values[0])] = 0.99
        my_nn.train(inputs, targets)
print("训练完毕")
# 保存训练好的权重数据
my_nn.wih.tofile("data/wih_trained.bin")
my_nn.who.tofile("data/who_trained.bin")
print("网络已保存")