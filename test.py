# -*- coding: utf-8 -*-
# File  : test.py
# Author: Meijin Lu
# Date  : 2020/10/23
import neural_network as nn
import numpy
import imageio
import matplotlib.pyplot


input_nodes = 784
hidden_nodes = 150
output_nodes = 10
learning_rate = 0.2
# 创建神经网络实例对象
# my_nn = nn.neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)
# my_nn.wih.tofile("wih_trained.bin")
# my_nn.who.tofile("who_trained.bin")

# wih_array = numpy.fromfile("wih_trained.bin", dtype=numpy.float64).reshape(150, 784)
# who_array = numpy.fromfile("who_trained.bin", dtype=numpy.float64).reshape(10, 150)
# my_nn.w_inputs(wih_array, who_array)
# print(my_nn.wih)

image_name = "手写数字识别/数字8.png"   # 图片路径
img_array = imageio.imread(image_name, as_gray=True)   # 得到的是二维矩阵
img_data = 255.0 - img_array.reshape(784)
query_inputs_list = img_data/255.0*0.99 + 0.01
inputs_array = numpy.asfarray(query_inputs_list).reshape(28, 28)
matplotlib.pyplot.imshow(inputs_array, cmap="Greys", interpolation="None")
matplotlib.pyplot.show()
