import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 数据准备 房价数据
housing = fetch_california_housing()
# 将总体分为训练集 验证集
X_train_full, X_test, y_train_full, y_test = train_test_split(
    housing.data, housing.target, random_state=42)
# 将训练集分为训练集 验证集
X_train, X_valid, y_train, y_valid = train_test_split(
    X_train_full, y_train_full, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)
# 1、fit
# 用于计算训练数据的均值和方差， 后面就会用均值和方差来转换训练数据
# 2、fit_transform
# 不仅计算训练数据的均值和方差，还会基于计算出来的均值和方差来转换训练数据，从而把数据转换成标准的正太分布
# 3、transform
# 很显然，它只是进行转换，只是把训练数据转换成标准的正态分布
np.random.seed(42)
tf.random.set_seed(42)
model = keras.models.Sequential([
    keras.layers.Dense(30, activation="relu",
                       input_shape=X_train.shape[1:]),  # 输入层 30个神经元
    keras.layers.Dense(1)  # 输出层为一个神经元
])
# 编译模型
model.compile(loss="mean_squared_error",
              optimizer=keras.optimizers.SGD(lr=1e-3))
# 模型训练
history = model.fit(X_train, y_train, epochs=30,
                    validation_data=(X_valid, y_valid))
# 模型评估
mse_test = model.evaluate(X_test, y_test)
print("评估", mse_test)  # 评估 0.42117786407470703
# 预测[[0.3885664], [1.6792021], [3.1022797]]
print("预测", model.predict(X_test[:3]))
# 显示迭代过程
plt.plot(pd.DataFrame(history.history))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.show()
print(X_test[:3])
print(y_test[:3])
# [[-1.15780104 -0.28673138 -0.49550877 -0.16618097 -0.02946012  0.38899735
#    0.19374821  0.2870474]
#  [-0.7125531   0.10880952 - 0.16332973  0.20164652  0.12842117 - 0.11818174
#   -0.23725261  0.06215231]
#  [-0.2156101   1.8491895  -0.57982788  0.18528489 -0.10429403 -0.67694905
#    1.00890193 -1.4271529 ]]
# [0.477   0.458   5.00001]
