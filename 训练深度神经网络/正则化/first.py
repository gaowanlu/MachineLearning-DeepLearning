# Avoiding Overfitting Through Regularization 通过正则化避免过度拟合
# ℓ1 and ℓ2 regularization l1与l2正则化
from scipy.special import erfc
import sklearn
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from functools import partial

# 优化器选择与优化
# 数据准备
(X_train_full, y_train_full), (X_test,
                               y_test) = keras.datasets.fashion_mnist.load_data()
X_train_full = X_train_full / 255.0
X_test = X_test / 255.0
X_valid, X_train = X_train_full[:5000], X_train_full[5000:]
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

# 设置l2超参数
layer = keras.layers.Dense(100, activation="elu",
                           kernel_initializer="he_normal",
                           kernel_regularizer=keras.regularizers.l2(0.01))
# 其他
# keras.regularizers.l2(0.1)
# keras.regularizers.l1(0.1)
# keras.regularizers.l1_l2(0.1, 0.01)
# 创建模型
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dense(300, activation="elu",
                       kernel_initializer="he_normal",
                       kernel_regularizer=keras.regularizers.l2(0.01)),
    keras.layers.Dense(100, activation="elu",
                       kernel_initializer="he_normal",
                       kernel_regularizer=keras.regularizers.l2(0.01)),
    keras.layers.Dense(10, activation="softmax",
                       kernel_regularizer=keras.regularizers.l2(0.01))
])
model.compile(loss="sparse_categorical_crossentropy",
              optimizer="nadam", metrics=["accuracy"])
# 训练
n_epochs = 2
history = model.fit(X_train, y_train, epochs=n_epochs,
                    validation_data=(X_valid, y_valid))

# 或者使用 functools 工具函数
# 在创建网络层时规定正则化参数
RegularizedDense = partial(keras.layers.Dense,
                           activation="elu",
                           kernel_initializer="he_normal",
                           kernel_regularizer=keras.regularizers.l2(0.01))
# 模型创建
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    RegularizedDense(300),  # 300 layers
    RegularizedDense(100),  # 100 layers
    RegularizedDense(10, activation="softmax")
])
# 模型编译
model.compile(loss="sparse_categorical_crossentropy",
              optimizer="nadam", metrics=["accuracy"])
# 模型训练
n_epochs = 2
history = model.fit(X_train, y_train, epochs=n_epochs,
                    validation_data=(X_valid, y_valid))


# Dropout 对于dnn最受欢迎得正则化技术之一由hinton提出
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dropout(rate=0.2),
    keras.layers.Dense(300, activation="elu", kernel_initializer="he_normal"),
    keras.layers.Dropout(rate=0.2),
    keras.layers.Dense(100, activation="elu", kernel_initializer="he_normal"),
    keras.layers.Dropout(rate=0.2),
    keras.layers.Dense(10, activation="softmax")
])
model.compile(loss="sparse_categorical_crossentropy",
              optimizer="nadam", metrics=["accuracy"])
n_epochs = 2
history = model.fit(X_train, y_train, epochs=n_epochs,
                    validation_data=(X_valid, y_valid))


# Alpha Dropout
# 如果要基于selu激活函数对自归一化网络进行正则化，则应使用alpha dropout,这是dropout的一种变体
# 它保留了其输入的均值和标准差
tf.random.set_seed(42)
np.random.seed(42)
# 模型创建 网络层有使用selu作为神经元激活函数，如需dropout需要使用alpha dropout
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.AlphaDropout(rate=0.2),
    keras.layers.Dense(300, activation="selu",
                       kernel_initializer="lecun_normal"),
    keras.layers.AlphaDropout(rate=0.2),
    keras.layers.Dense(100, activation="selu",
                       kernel_initializer="lecun_normal"),
    keras.layers.AlphaDropout(rate=0.2),
    keras.layers.Dense(10, activation="softmax")
])
optimizer = keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True)
model.compile(loss="sparse_categorical_crossentropy",
              optimizer=optimizer, metrics=["accuracy"])
# 训练
n_epochs = 20
history = model.fit(X_train, y_train, epochs=n_epochs,
                    validation_data=(X_valid, y_valid))
# 评估
model.evaluate(X_test, y_test)
model.evaluate(X_train, y_train)
# 再训练
history = model.fit(X_train, y_train)


# MC Dropout
# 蒙特卡洛方法 先了解知道有这个 学到理论内容或者需要时再看吧
tf.random.set_seed(42)
np.random.seed(42)
y_probas = np.stack([model(X_test, training=True)
                     for sample in range(100)])
y_proba = y_probas.mean(axis=0)
y_std = y_probas.std(axis=0)
np.round(model.predict(X_test[:1]), 2)
np.round(y_probas[:, :1], 2)
np.round(y_proba[:1], 2)
y_std = y_probas.std(axis=0)
np.round(y_std[:1], 2)
y_pred = np.argmax(y_proba, axis=1)
accuracy = np.sum(y_pred == y_test) / len(y_test)
accuracy


class MCDropout(keras.layers.Dropout):
    def call(self, inputs):
        return super().call(inputs, training=True)


class MCAlphaDropout(keras.layers.AlphaDropout):
    def call(self, inputs):
        return super().call(inputs, training=True)


# seed
tf.random.set_seed(42)
np.random.seed(42)
# 模型创建 使用Alpha Dropout部分处的模型 如果层为AlphaDropout则替换为MCDropout
mc_model = keras.models.Sequential([
    MCAlphaDropout(layer.rate) if isinstance(
        layer, keras.layers.AlphaDropout) else layer
    for layer in model.layers
])
mc_model.summary()
optimizer = keras.optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True)
mc_model.compile(loss="sparse_categorical_crossentropy",
                 optimizer=optimizer, metrics=["accuracy"])
mc_model.set_weights(model.get_weights())
np.round(np.mean([mc_model.predict(X_test[:1])
                  for sample in range(100)], axis=0), 2)
# Max norm
layer = keras.layers.Dense(100, activation="selu", kernel_initializer="lecun_normal",
                           kernel_constraint=keras.constraints.max_norm(1.))
MaxNormDense = partial(keras.layers.Dense,
                       activation="selu", kernel_initializer="lecun_normal",
                       kernel_constraint=keras.constraints.max_norm(1.))

model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    MaxNormDense(300),
    MaxNormDense(100),
    keras.layers.Dense(10, activation="softmax")
])
model.compile(loss="sparse_categorical_crossentropy",
              optimizer="nadam", metrics=["accuracy"])
n_epochs = 2
history = model.fit(X_train, y_train, epochs=n_epochs,
                    validation_data=(X_valid, y_valid))
