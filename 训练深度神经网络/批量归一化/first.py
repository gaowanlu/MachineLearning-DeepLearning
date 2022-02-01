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

# 使用批量归一化层
'''
该操作对每个输入零中心并归一化，然后每层使用两个新的参数向量
缩放和偏移其结果：一个用于缩放，另一个用于偏移。换句话说，该操作可以使模型学习
各层输入的最佳缩放和均值。在许多情况下，如果你将BN层添加为神经网络的第一层，
则无须归一化训练集（例如，使用StandardScaler）；BN层会为你完成此操作（因为它一
次只能查看一个批次，它还可以重新缩放和偏移每个输入特征）
'''
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(300, activation="relu"),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(100, activation="relu"),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(10, activation="softmax")
])
bn1 = model.layers[1]
print([(var.name, var.trainable) for var in bn1.variables])
print(bn1.updates)
model.compile(loss="sparse_categorical_crossentropy",
              optimizer=keras.optimizers.SGD(lr=1e-3),
              metrics=["accuracy"])
history = model.fit(X_train, y_train, epochs=10,
                    validation_data=(X_valid, y_valid))
'''
有时，在激活函数工作得更好之前应用BN（关于这个话题存在争议）。
此外，“BatchNormalization”层之前的层不需要有偏差项，
因为“BatchNormalization”层也有偏差项，这会浪费参数，
所以在创建这些层时可以设置“use_bias=False”：
'''
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(300, use_bias=False),
    keras.layers.BatchNormalization(),
    keras.layers.Activation("relu"),  # "relu" tf.nn.relu \tf.nn.elu ......
    keras.layers.Dense(100, use_bias=False),
    keras.layers.BatchNormalization(),
    keras.layers.Activation("relu"),
    keras.layers.Dense(10, activation="softmax")
])

model.compile(loss="sparse_categorical_crossentropy",
              optimizer=keras.optimizers.SGD(lr=1e-3),
              metrics=["accuracy"])
history = model.fit(X_train, y_train, epochs=10,
                    validation_data=(X_valid, y_valid))
print(np.argmax(model.predict(X_test[:3]), axis=-1))  # [9 2 1]
