# -*- coding: utf-8 -*-
# File  : neural_network.py
# Author: Meijin Lu
# Date  : 2020/10/21
import numpy
import scipy.special


class neuralNetwork:
    def __init__(self, inputnodes, hiddennodes, outputsnodes, learningrate):
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputsnodes
        self.lr = learningrate
        # 创建权重矩阵
        self.wih = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = numpy.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))
        # S激活函数
        self.activation_function = lambda x: scipy.special.expit(x)

    def train(self, inputs_list, targets_list):
        # 数据格式化
        inputs = numpy.array(inputs_list, ndmin=2).T  # 至少二维矩阵
        targets = numpy.array(targets_list, ndmin=2).T

        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        output_errors = targets - final_outputs
        hidden_errors = numpy.dot(self.who.T, output_errors)
        self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)),
                                        numpy.transpose(hidden_outputs))
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),
                                        numpy.transpose(inputs))

    def query(self, inputs_list):
        inputs = numpy.array(inputs_list, ndmin=2).T  # 至少二维矩阵
        # print("其中的输入矩阵为:", inputs.T)
        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        return final_outputs

    def w_inputs(self, wih_array, who_array):
        self.wih = wih_array
        self.who = who_array


if __name__ == '__main__':
    input_nodes = 784
    hidden_nodes = 100
    output_nodes = 10
    learning_rate = 0.3
    # 创建神经网络实例对象
    my_nn = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)
    # 从csv文件加载训练所需数据
    training_data_file = open("data/mnist_train_100.csv", "r")
    training_data_list = training_data_file.readlines()     # training_data_list类似于["1,2","3,4"]
    training_data_file.close()
    # 开始训练
    for record in training_data_list:
        all_values = record.split(",")      # all_values每个元素以逗号分隔,如['1', '0', '0', '0', '0']
        # inputs每个元素以空格分隔,如[1 0 0
        #                            0 0 1]
        inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        targets = numpy.zeros(output_nodes) + 0.01
        targets[int(all_values[0])] = 0.99
        my_nn.train(inputs, targets)
    print("训练完毕")

    test_data_file = open("data/mnist_test_10.csv", "r")
    test_data_list = test_data_file.readlines()  # test_data_list为以每一行为元素的一维列表
    test_data_file.close()
    # print("test_data_list[0]是", test_data_list[0])
    scorecard = []
    for record in test_data_list:
        all_values = record.split(",")
        correct_label = int(record[0])
        # print("标签值为", all_values[0])
        # 画出图像
        # image_array = numpy.asfarray(all_values[1:]).reshape(28, 28)
        # mp.imshow(image_array, cmap="Greys", interpolation="None")
        # mp.show()
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
    # scorecard是[1, 1, 1, 1, 1, 1, 0, 0, 0, 0]
    # scorecard_array是
    # [1
    # 1
    # 1
    # 1
    # 1
    # 1
    # 0
    # 0
    # 0
    # 0]
    print("准确率为:", scorecard_array.sum() / scorecard_array.size)
