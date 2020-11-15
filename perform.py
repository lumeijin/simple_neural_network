# -*- coding: utf-8 -*-
# File  : perform.py
# Author: Meijin Lu
# Date  : 2020/10/21
import neural_network as nn
import numpy

input_nodes = 784
hidden_nodes = 200
output_nodes = 10
learning_rate = 0.1
# 创建神经网络实例对象
my_nn = nn.neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)
# 从csv文件加载训练所需数据
training_data_file = open("data/mnist_train.csv", "r")
training_data_list = training_data_file.readlines()
training_data_file.close()
# 开始训练
epoches = 5  # 世代
for e in range(epoches):
    for record in training_data_list:
        all_values = record.split(",")
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        targets = numpy.zeros(output_nodes) + 0.01
        targets[int(all_values[0])] = 0.99
        my_nn.train(inputs, targets)
print("训练完毕")

test_data_file = open("data/mnist_test.csv", "r")
test_data_list = test_data_file.readlines()  # test_data_list为以每一行为元素的一维列表
test_data_file.close()

scorecard = []
for record in test_data_list:
    all_values = record.split(",")
    correct_label = int(record[0])
    # 查询
    query_inputs_list = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99 + 0.01)  # 得到一个列表
    query_results = my_nn.query(query_inputs_list)
    # print("query_results是", query_results)
    query_label = numpy.argmax(query_results)  # 得到最大值的索引值
    if query_label == correct_label:
        scorecard.append(1)
    else:
        scorecard.append(0)
# 计算准确率
scorecard_array = numpy.asarray(scorecard)
print("准确率为:", scorecard_array.sum() / scorecard_array.size)
