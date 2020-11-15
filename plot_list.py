import matplotlib.pyplot
import numpy

a = numpy.zeros([3, 2])
a[0, 0] = 1
a[0, 1] = 2
a[1, 0] = 9
a[2, 1] = 12
print("a的值是", a)
matplotlib.pyplot.imshow(a, interpolation="nearest")
matplotlib.pyplot.show()  # 在非交互式环境中请务必加上这一句
