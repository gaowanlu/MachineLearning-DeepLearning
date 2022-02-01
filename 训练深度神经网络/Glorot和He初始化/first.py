import sklearn
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

'''
想要训练出一个好的模型 我们要有许多指标以及数据的预处理的动作要做 以及可能途中遇见很多问题
'''
# 对激活函数的探讨与注意事项

# Sigmoid 激活函数


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


z = np.linspace(-5, 5, 200)  # 线性空间 [-5 5] 200个特征
plt.plot([-5, 5], [0, 0], 'k-')
plt.plot([-5, 5], [1, 1], 'k--')
plt.plot([0, 0], [-0.2, 1.2], 'k-')
plt.plot([-5, 5], [-3/4, 7/4], 'g--')
plt.plot(z, sigmoid(z), "b-", linewidth=2)
props = dict(facecolor='black', shrink=0.1)
plt.grid(True)
plt.title("Sigmoid activation function", fontsize=14)
plt.axis([-5, 5, -0.2, 1.2])
plt.show()


# 权重初始化方法Xavier，He initialization
# 在每一层的不同的神经元可能需要不同的权重比 但 初始值怎么设定呢 科学家们也在不断地总结出经验
print([name for name in dir(keras.initializers) if not name.startswith("_")])
'''
keras.initializers._____ public
['Constant', 'GlorotNormal', 'GlorotUniform', 'HeNormal', 
'HeUniform', 'Identity', 'Initializer', 'LecunNormal',
 'LecunUniform', 'Ones', 'Orthogonal', 'RandomNormal',
  'RandomUniform', 'TruncatedNormal', 'VarianceScaling',
   'Zeros', 'constant', 'deserialize', 'get', 'glorot_normal', 
   'glorot_uniform', 'he_normal', 'he_uniform', 'identity', 
   'lecun_normal', 'lecun_uniform', 'ones', 'orthogonal', 
   'random_normal', 'random_uniform', 'serialize', 
   'truncated_normal', 'variance_scaling', 'zeros']
'''
# 默认情况keras选择具有均匀分布的Glorot初始化 可以通过 kernel_initializer="he_normal" 进行更改为He初始化
keras.layers.Dense(10, activation="relu", kernel_initializer="he_normal")

# 如果使用均匀分布 但基于fanavg 进行 He初始化使用VarianceScaling
init = keras.initializers.VarianceScaling(scale=2., mode='fan_avg',
                                          distribution='uniform')
keras.layers.Dense(10, activation="relu", kernel_initializer=init)

'''            激活函数                                      segama^2
Glorot          None\tanh\logistic\softmax                     1/fanavg
He              ReLU和变体                                      2/fanin
LeCun           RELU                                           1/fanin
'''
