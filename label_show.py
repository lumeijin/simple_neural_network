# -*- coding: utf-8 -*-
# File  : label_show.py
# Author: Meijin Lu
# Date  : 2020/10/21
import numpy
import matplotlib.pyplot as mp
# data_file = open("data/mnist_train_100.csv", "r")
data_file = open("data/mnist_test_10.csv", "r")
data_list = data_file.readlines()
data_file.close()
# 画出第几个图,从0-99或0-9
index = 4
print("标签值为:", data_list[index][0])
all_values = data_list[index].split(",")
image_array = numpy.asfarray(all_values[1:]).reshape(28, 28)
mp.imshow(image_array, cmap="Greys", interpolation="None")
mp.show()