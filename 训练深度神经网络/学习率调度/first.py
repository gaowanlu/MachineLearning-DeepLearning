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


# 学习率调度Learning Rate Scheduling
# -------------------------------------------------------
# Power Scheduling # 幂调度 设置超参数 decay
# 以高学习率开始 然后衰减 比较好
'''
```lr = lr0 / (1 + steps / s)**c```
* Keras uses `c = 1` and `s = 1 / decay`
'''


def section_1():
    optimizer = keras.optimizers.SGD(lr=0.01, decay=1e-4)
    model = keras.models.Sequential([
        keras.layers.Flatten(input_shape=[28, 28]),
        keras.layers.Dense(300, activation="selu",
                           kernel_initializer="lecun_normal"),
        keras.layers.Dense(100, activation="selu",
                           kernel_initializer="lecun_normal"),
        keras.layers.Dense(10, activation="softmax")
    ])
    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer=optimizer, metrics=["accuracy"])
    n_epochs = 25
    history = model.fit(X_train, y_train, epochs=n_epochs,
                        validation_data=(X_valid, y_valid))
    learning_rate = 0.01
    decay = 1e-4
    batch_size = 32
    n_steps_per_epoch = len(X_train) // batch_size
    epochs = np.arange(n_epochs)
    lrs = learning_rate / (1 + decay * epochs * n_steps_per_epoch)

    plt.plot(epochs, lrs,  "o-")
    plt.axis([0, n_epochs - 1, 0, 0.01])
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.title("Power Scheduling", fontsize=14)
    plt.grid(True)
    plt.show()


# section_1()

# -------------------------------------------------------
# Exponential Scheduling指数调度


'''
```lr = lr0 * 0.1**(epoch / s)```
'''


def section_2():

    def exponential_decay(lr0, s):
        def exponential_decay_fn(epoch):
            return lr0 * 0.1**(epoch / s)
        return exponential_decay_fn

    # 学习率0.01步20
    exponential_decay_fn = exponential_decay(lr0=0.01, s=20)

    # 创建模型
    model = keras.models.Sequential([
        keras.layers.Flatten(input_shape=[28, 28]),
        keras.layers.Dense(300, activation="selu",
                           kernel_initializer="lecun_normal"),
        keras.layers.Dense(100, activation="selu",
                           kernel_initializer="lecun_normal"),
        keras.layers.Dense(10, activation="softmax")
    ])
    # 编译模型
    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer="nadam", metrics=["accuracy"])
    n_epochs = 25
    # 学习率回调函数参数为epoch 需返回实时学习率
    lr_scheduler = keras.callbacks.LearningRateScheduler(exponential_decay_fn)
    history = model.fit(X_train, y_train, epochs=n_epochs,
                        validation_data=(X_valid, y_valid),
                        callbacks=[lr_scheduler])
    # 画出学习率变化
    plt.plot(history.epoch, history.history["lr"], "o-")
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.grid(True)
    plt.show()

# section_2()

# --------------------------------------------------------------
# Keras方法指数调度
# 继承keras.callbacks.Callback


def section_3():
    class ExponentialDecay(keras.callbacks.Callback):
        def __init__(self, s=40000):
            super().__init__()
            self.s = s

        # 在每次epoch开始前设置最新学习率
        def on_batch_begin(self, batch, logs=None):
            # Note: the `batch` argument is reset at each epoch
            lr = keras.backend.get_value(self.model.optimizer.lr)
            keras.backend.set_value(self.model.optimizer.lr, lr * 0.1**(1 / s))

        # 在每次epoch结束后记录学习率

        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            logs['lr'] = keras.backend.get_value(self.model.optimizer.lr)

    # 创建模型
    model = keras.models.Sequential([
        keras.layers.Flatten(input_shape=[28, 28]),
        keras.layers.Dense(300, activation="selu",
                           kernel_initializer="lecun_normal"),
        keras.layers.Dense(100, activation="selu",
                           kernel_initializer="lecun_normal"),
        keras.layers.Dense(10, activation="softmax")
    ])
    lr0 = 0.01  # 初始学习率
    optimizer = keras.optimizers.Nadam(lr=lr0)
    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer=optimizer, metrics=["accuracy"])
    n_epochs = 25  # 总epoch数

    # number of steps in 20 epochs (batch size = 32)
    s = 20 * len(X_train) // 32
    # 创建自定义callback实例
    exp_decay = ExponentialDecay(s)
    history = model.fit(X_train, y_train, epochs=n_epochs,
                        validation_data=(X_valid, y_valid),
                        callbacks=[exp_decay])
    # 总步数
    n_steps = n_epochs * len(X_train) // 32
    steps = np.arange(n_steps)  # X坐标序列
    lrs = lr0 * 0.1**(steps / s)  # 学习率变化
    plt.plot(steps, lrs, "-", linewidth=2)
    plt.xlabel("Batch")
    plt.ylabel("Learning Rate")
    plt.title("Exponential Scheduling (per batch)", fontsize=14)
    plt.grid(True)
    plt.show()


# section_3()


# ----------------------------------------------------------------
# 分段常数调度Piecewise Constant Scheduling 顾名思义

def section_4():
    def piecewise_constant(boundaries, values):
        boundaries = np.array([0] + boundaries)
        values = np.array(values)

        def piecewise_constant_fn(epoch):
            return values[np.argmax(boundaries > epoch) - 1]
        return piecewise_constant_fn

    # 分段
    piecewise_constant_fn = piecewise_constant([5, 15], [0.01, 0.005, 0.001])
    lr_scheduler = keras.callbacks.LearningRateScheduler(piecewise_constant_fn)
    # 模型创建
    model = keras.models.Sequential([
        keras.layers.Flatten(input_shape=[28, 28]),
        keras.layers.Dense(300, activation="selu",
                           kernel_initializer="lecun_normal"),
        keras.layers.Dense(100, activation="selu",
                           kernel_initializer="lecun_normal"),
        keras.layers.Dense(10, activation="softmax")
    ])
    # 模型编译
    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer="nadam", metrics=["accuracy"])
    # 训练
    n_epochs = 25
    history = model.fit(X_train, y_train, epochs=n_epochs,
                        validation_data=(X_valid, y_valid),
                        callbacks=[lr_scheduler])

    plt.plot(history.epoch, [piecewise_constant_fn(epoch)
                             for epoch in history.epoch], "o-")
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.title("Piecewise Constant Scheduling", fontsize=14)
    plt.grid(True)
    plt.show()


# section_4()


# --------------------------------------------------------------
# Performance Scheduling性能调度
# 每当使用5个轮次的最好验证损失都没有改善时，它将使学习率乘以0.5
def section_5():
    tf.random.set_seed(42)
    np.random.seed(42)
    lr_scheduler = keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
    # 模型创建
    model = keras.models.Sequential([
        keras.layers.Flatten(input_shape=[28, 28]),
        keras.layers.Dense(300, activation="selu",
                           kernel_initializer="lecun_normal"),
        keras.layers.Dense(100, activation="selu",
                           kernel_initializer="lecun_normal"),
        keras.layers.Dense(10, activation="softmax")
    ])
    optimizer = keras.optimizers.SGD(lr=0.02, momentum=0.9)
    # 编译
    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer=optimizer, metrics=["accuracy"])
    # 训练
    n_epochs = 25
    history = model.fit(X_train, y_train, epochs=n_epochs,
                        validation_data=(X_valid, y_valid),
                        callbacks=[lr_scheduler])
    # 画出学习率
    plt.plot(history.epoch, history.history["lr"], "bo-")
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate", color='b')
    plt.tick_params('y', colors='b')
    plt.gca().set_xlim(0, n_epochs - 1)
    plt.grid(True)
    # 画出验证集损失
    ax2 = plt.gca().twinx()
    ax2.plot(history.epoch, history.history["val_loss"], "r^-")
    ax2.set_ylabel('Validation Loss', color='r')
    ax2.tick_params('y', colors='r')

    plt.title("Reduce LR on Plateau", fontsize=14)
    plt.show()


# section_5()

# -----------------------------------------------------------


# keras 优化器设置学习率 怎么用都可以 它们提供了形如下面的API
# keras.optimizers.schedules.PiecewiseConstantDecay ... soon
def section_6():
    model = keras.models.Sequential([
        keras.layers.Flatten(input_shape=[28, 28]),
        keras.layers.Dense(300, activation="selu",
                           kernel_initializer="lecun_normal"),
        keras.layers.Dense(100, activation="selu",
                           kernel_initializer="lecun_normal"),
        keras.layers.Dense(10, activation="softmax")
    ])
    # number of steps in 20 epochs (batch size = 32)
    s = 20 * len(X_train) // 32
    learning_rate = keras.optimizers.schedules.ExponentialDecay(0.01, s, 0.1)
    optimizer = keras.optimizers.SGD(learning_rate)
    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer=optimizer, metrics=["accuracy"])
    n_epochs = 25
    history = model.fit(X_train, y_train, epochs=n_epochs,
                        validation_data=(X_valid, y_valid))
    batch_size = 32
    n_steps_per_epoch = len(X_train) // batch_size
    # 如 分段常量调度
    learning_rate = keras.optimizers.schedules.PiecewiseConstantDecay(
        boundaries=[5. * n_steps_per_epoch, 15. * n_steps_per_epoch],
        values=[0.01, 0.005, 0.001])

# -----------------------------------------------------------------------

# 1Cycle scheduling 1周期调度
# 用到再学吧 实在太多了

# 综合使用 指数调度为例


def section_7():
    class ExponentialLearningRate(keras.callbacks.Callback):
        def __init__(self, factor):
            self.factor = factor
            self.rates = []
            self.losses = []

        def on_batch_end(self, batch, logs):
            # 记录学习率
            self.rates.append(keras.backend.get_value(self.model.optimizer.lr))
            # 记录损失值
            self.losses.append(logs["loss"])
            # 设置学习率
            keras.backend.set_value(self.model.optimizer.lr,
                                    self.model.optimizer.lr * self.factor)

    def find_learning_rate(model, X, y, epochs=1, batch_size=32, min_rate=10**-5, max_rate=10):
        init_weights = model.get_weights()
        iterations = len(X) // batch_size * epochs
        factor = np.exp(np.log(max_rate / min_rate) / iterations)
        init_lr = keras.backend.get_value(model.optimizer.lr)
        keras.backend.set_value(model.optimizer.lr, min_rate)
        exp_lr = ExponentialLearningRate(factor)
        history = model.fit(X, y, epochs=epochs, batch_size=batch_size,
                            callbacks=[exp_lr])
        keras.backend.set_value(model.optimizer.lr, init_lr)
        model.set_weights(init_weights)
        return exp_lr.rates, exp_lr.losses

    def plot_lr_vs_loss(rates, losses):
        plt.plot(rates, losses)
        plt.gca().set_xscale('log')
        plt.hlines(min(losses), min(rates), max(rates))
        plt.xlabel("Learning rate")
        plt.ylabel("Loss")

    tf.random.set_seed(42)
    np.random.seed(42)
    # 创建模型
    model = keras.models.Sequential([
        keras.layers.Flatten(input_shape=[28, 28]),
        keras.layers.Dense(300, activation="selu",
                           kernel_initializer="lecun_normal"),
        keras.layers.Dense(100, activation="selu",
                           kernel_initializer="lecun_normal"),
        keras.layers.Dense(10, activation="softmax")
    ])
    # 编译
    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer=keras.optimizers.SGD(lr=1e-3),
                  metrics=["accuracy"])
    # 训练
    batch_size = 128
    rates, losses = find_learning_rate(
        model, X_train, y_train, epochs=1, batch_size=batch_size)
    # 画出学习率与损失值得变化
    plot_lr_vs_loss(rates, losses)
    plt.show()


section_7()

# ------------------------------------------------------------------
# 1周期调度内容


def section_8():
    lr_rates = []

    class OneCycleScheduler(keras.callbacks.Callback):
        def __init__(self, iterations, max_rate, start_rate=None,
                     last_iterations=None, last_rate=None):
            self.iterations = iterations
            self.max_rate = max_rate
            self.start_rate = start_rate or max_rate / 10
            self.last_iterations = last_iterations or iterations // 10 + 1
            self.half_iteration = (iterations - self.last_iterations) // 2
            self.last_rate = last_rate or self.start_rate / 1000
            self.iteration = 0

        def _interpolate(self, iter1, iter2, rate1, rate2):
            return ((rate2 - rate1) * (self.iteration - iter1)
                    / (iter2 - iter1) + rate1)

        def on_batch_begin(self, batch, logs):
            if self.iteration < self.half_iteration:
                rate = self._interpolate(
                    0, self.half_iteration, self.start_rate, self.max_rate)
            elif self.iteration < 2 * self.half_iteration:
                rate = self._interpolate(self.half_iteration, 2 * self.half_iteration,
                                         self.max_rate, self.start_rate)
            else:
                rate = self._interpolate(2 * self.half_iteration, self.iterations,
                                         self.start_rate, self.last_rate)
                rate = max(rate, self.last_rate)
            self.iteration += 1
            keras.backend.set_value(self.model.optimizer.lr, rate)
            lr_rates.append(keras.backend.get_value(self.model.optimizer.lr))
    # 创建模型
    model = keras.models.Sequential([
        keras.layers.Flatten(input_shape=[28, 28]),
        keras.layers.Dense(300, activation="selu",
                           kernel_initializer="lecun_normal"),
        keras.layers.Dense(100, activation="selu",
                           kernel_initializer="lecun_normal"),
        keras.layers.Dense(10, activation="softmax")
    ])
    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer=keras.optimizers.SGD(lr=1e-3),
                  metrics=["accuracy"])
    n_epochs = 25
    batch_size = 128
    onecycle = OneCycleScheduler(
        len(X_train) // batch_size * n_epochs, max_rate=0.05)
    history = model.fit(X_train, y_train, epochs=n_epochs, batch_size=batch_size,
                        validation_data=(X_valid, y_valid),
                        callbacks=[onecycle])
    print("LR_RATES_SIZE", len(lr_rates))
    steps = np.arange(len(lr_rates))  # X坐标序列
    plt.plot(steps, lr_rates, "-", linewidth=2)
    plt.xlabel("batch")
    plt.ylabel("lr")
    plt.grid(True)
    plt.show()


section_8()
