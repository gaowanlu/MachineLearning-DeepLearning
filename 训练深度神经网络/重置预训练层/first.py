# Reusing Pretrained Layers
# 重置预训练层 主要运用与模型的修改 找一个相似的神经网络 然后使用网络的较低层 此技术成为迁移学习
import os
from scipy.special import erfc
import sklearn
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

# 数据准备
(X_train_full, y_train_full), (X_test,
                               y_test) = keras.datasets.fashion_mnist.load_data()
X_train_full = X_train_full / 255.0
X_test = X_test / 255.0
X_valid, X_train = X_train_full[:5000], X_train_full[5000:]
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

# 数据集预处理


def split_dataset(X, y):
    y_5_or_6 = (y == 5) | (y == 6)  # sandals or shirts
    y_A = y[~y_5_or_6]
    y_A[y_A > 6] -= 2  # class indices 7, 8, 9 should be moved to 5, 6, 7
    # binary classification task: is it a shirt (class 6)?
    y_B = (y[y_5_or_6] == 6).astype(np.float32)
    return ((X[~y_5_or_6], y_A),
            (X[y_5_or_6], y_B))


(X_train_A, y_train_A), (X_train_B, y_train_B) = split_dataset(X_train, y_train)
(X_valid_A, y_valid_A), (X_valid_B, y_valid_B) = split_dataset(X_valid, y_valid)
(X_test_A, y_test_A), (X_test_B, y_test_B) = split_dataset(X_test, y_test)
X_train_B = X_train_B[:200]
y_train_B = y_train_B[:200]


print(X_train_A.shape)
print(X_train_B.shape)
tf.random.set_seed(42)
np.random.seed(42)

# 创建模型A
model_A = keras.models.Sequential()
model_A.add(keras.layers.Flatten(input_shape=[28, 28]))
# 添加5个隐藏层
for n_hidden in (300, 100, 50, 50, 50):
    model_A.add(keras.layers.Dense(n_hidden, activation="selu"))
# 输出层
model_A.add(keras.layers.Dense(8, activation="softmax"))
model_A.compile(loss="sparse_categorical_crossentropy",
                optimizer=keras.optimizers.SGD(lr=1e-3),
                metrics=["accuracy"])
history = model_A.fit(X_train_A, y_train_A, epochs=20,
                      validation_data=(X_valid_A, y_valid_A))


# 模型B
model_B = keras.models.Sequential()
model_B.add(keras.layers.Flatten(input_shape=[28, 28]))
for n_hidden in (300, 100, 50, 50, 50):
    model_B.add(keras.layers.Dense(n_hidden, activation="selu"))
# 输出层为一个神经元 激活函数为sigmoid
model_B.add(keras.layers.Dense(1, activation="sigmoid"))
model_B.compile(loss="binary_crossentropy",
                optimizer=keras.optimizers.SGD(lr=1e-3),
                metrics=["accuracy"])
history = model_B.fit(X_train_B, y_train_B, epochs=20,
                      validation_data=(X_valid_B, y_valid_B))


# 将modelA的所有层重新Sequential 但排除最后一个输出层
model_B_on_A = keras.models.Sequential(model_A.layers[:-1])
# 添加一个输出层 一个神经元 激活函数为 sigmoid
model_B_on_A.add(keras.layers.Dense(1, activation="sigmoid"))
# 克隆模型 ：为了放置共享层 在训练modelBonA时会修改modelA 要保存A 则在训练前复制A 留下副本
model_A_clone = keras.models.clone_model(model_A)
# 获取权重 设置权重 clone模型并不会复制权重 需要手动赋值
model_A_clone.set_weights(model_A.get_weights())

# 遍历所有层 将禁用可训练修改 训练几轮以便新加入的层可以使用调整 做出合适的权重
for layer in model_B_on_A.layers[:-1]:
    layer.trainable = False
# 编译迁移后的模型
model_B_on_A.compile(loss="binary_crossentropy",
                     optimizer=keras.optimizers.SGD(lr=1e-3),
                     metrics=["accuracy"])

# 使用修改后的模型训练几轮
history = model_B_on_A.fit(X_train_B, y_train_B, epochs=4,
                           validation_data=(X_valid_B, y_valid_B))
# 新的层会适应就层的权重分布 将所有层设置为可训练
for layer in model_B_on_A.layers[:-1]:
    layer.trainable = True

# 编译模型
model_B_on_A.compile(loss="binary_crossentropy",
                     optimizer=keras.optimizers.SGD(lr=1e-3),
                     metrics=["accuracy"])
# 在正式使用修改后的模型进行训练
history = model_B_on_A.fit(X_train_B, y_train_B, epochs=16,
                           validation_data=(X_valid_B, y_valid_B))
print("modelB", model_B.evaluate(X_test_B, y_test_B))
print("modelB_ON_modelA", model_B_on_A.evaluate(X_test_B, y_test_B))
print("model train end")
