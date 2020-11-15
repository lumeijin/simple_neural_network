# -*- coding: utf-8 -*-
# File  : recognize_handwritten_numbers.py
# Author: Meijin Lu
# Date  : 2020/10/24
import neural_network as nn
import numpy
import imageio   # 参考https://imageio.readthedocs.io/en/latest/scipy.html

# 注意和obtain_the_trained_network.py中的数据保持一致
input_nodes = 784
hidden_nodes = 150
output_nodes = 10
learning_rate = 0.15
# 导入保存好的网络权重
my_nn = nn.neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)
wih_array = numpy.fromfile("data/wih_trained.bin", dtype=numpy.float64).reshape(my_nn.hnodes, my_nn.inodes)
who_array = numpy.fromfile("data/who_trained.bin", dtype=numpy.float64).reshape(my_nn.onodes, my_nn.hnodes)
my_nn.w_inputs(wih_array, who_array)
# 计算准确率
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
scorecard_array = numpy.asarray(scorecard)
print("准确率为:", scorecard_array.sum() / scorecard_array.size)
# 开始识别
image_name = "手写数字识别/数字4.png"   # 图片路径
img_array = imageio.imread(image_name, as_gray=True)   # 得到的是二维矩阵
img_data = 255.0 - img_array.reshape(784)
query_inputs_list = img_data/255.0*0.99 + 0.01
query_results = my_nn.query(query_inputs_list)
print("query_results是:\n", query_results)
query_label = numpy.argmax(query_results)   # 得到最大值的索引值
print("识别出的数字为:", query_label)