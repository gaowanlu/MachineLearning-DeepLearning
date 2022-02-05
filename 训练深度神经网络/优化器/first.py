import os
from scipy.special import erfc
import sklearn
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
# 优化器选择与优化
# 数据准备
(X_train_full, y_train_full), (X_test,
                               y_test) = keras.datasets.fashion_mnist.load_data()
X_train_full = X_train_full / 255.0
X_test = X_test / 255.0
X_valid, X_train = X_train_full[:5000], X_train_full[5000:]
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]


# momentum 动量优化  描述：球在挂画表面上沿平缓的坡度滚动，
# 开始速度慢、很快获得动量 知道到达终极速度
# 0.9 在实际运用往往表现得很好
# 可有加速收敛的效果
optimizer = keras.optimizers.SGD(lr=0.001, momentum=0.9)

# nesterov加速梯度往往比动量优化更快
# nesterov accelerated gradient NAG
optimizer = keras.optimizers.SGD(lr=0.001, momentum=0.9, nesterov=True)

# AdaGrad与梯度下降相比 它通过沿最陡峭的维度按比例缩小梯度向，更多的指向全局最优解
optimizer = keras.optimizers.Adagrad(lr=0.001)
# AdaGrad有下降太快不收敛的风险
# RMSProp算法通过只是累计最近迭代中的梯度 不是自训练开始以来的1所有梯度
# 来解决此问题 rho为衰减率 该优化器通常比 adagrad好
optimizer = keras.optimizers.RMSprop(lr=0.001, rho=0.9)

# adam结合RMsprop与动量优化 像动量优化一样追踪过去的梯度衰减平均值
# beta1动量衰减系数 beta2缩放衰减系数
optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999)
# Adamax
optimizer = keras.optimizers.Adamax(lr=0.001, beta_1=0.9, beta_2=0.999)
# Nadam adam优化加上nesterov技巧
optimizer = keras.optimizers.Nadam(lr=0.001, beta_1=0.9, beta_2=0.999)
