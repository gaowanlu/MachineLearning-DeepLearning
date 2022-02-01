import os
from scipy.special import erfc
import sklearn
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

z = np.linspace(-5, 5, 200)  # 线性空间 [-5 5] 200个特征

(X_train_full, y_train_full), (X_test,
                               y_test) = keras.datasets.fashion_mnist.load_data()
X_train_full = X_train_full / 255.0
X_test = X_test / 255.0
X_valid, X_train = X_train_full[:5000], X_train_full[5000:]
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

# sigmoid和tanh是“饱和激活函数”,而ReLU及其变体则是“非饱和激活函数”
# leaky_relu激活函数

# 画leaky_relu图像


def part_1():
    def leaky_relu(z, alpha=0.01):
        return np.maximum(alpha*z, z)
    y = leaky_relu(z, 0.05)
    print(y)
    plt.plot(z, y, "b-", linewidth=2)
    plt.plot([-5, 5], [0, 0], 'k-')
    plt.plot([0, 0], [-0.5, 4.2], 'k-')
    plt.grid(True)
    plt.title("Leaky ReLU activation function", fontsize=14)
    plt.show()

    print([m for m in dir(keras.activations) if not m.startswith("_")])
    '''keras 激活函数
    ['deserialize', 'elu', 'exponential', 'get', 'hard_sigmoid', 'linear', 'relu',
        'selu', 'serialize', 'sigmoid', 'softmax', 'softplus', 'softsign', 'swish', 'tanh']
    '''

    # ReLU及其变体
    print([m for m in dir(keras.layers) if "relu" in m.lower()])
    # ['LeakyReLU', 'PReLU', 'ReLU', 'ThresholdedReLU']
    # keras.layers.LeakyReLU


# - -------------------------------Leaky ReLU-------------------------------
# 使用Leaky ReLU
def part_2():

    tf.random.set_seed(42)
    np.random.seed(42)

    model = keras.models.Sequential([
        keras.layers.Flatten(input_shape=[28, 28]),
        keras.layers.Dense(300, kernel_initializer="he_normal"),
        keras.layers.LeakyReLU(),
        keras.layers.Dense(100, kernel_initializer="he_normal"),
        keras.layers.LeakyReLU(),
        keras.layers.Dense(10, activation="softmax")
    ])
    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer=keras.optimizers.SGD(lr=1e-3),
                  metrics=["accuracy"])
    # 指标指定为精准率
    history = model.fit(X_train, y_train, epochs=10,
                        validation_data=(X_valid, y_valid))
    print("预测", model.predict_classes(X_test[:3]))
    '''
    Leaky version of a Rectified Linear Unit.

    It allows a small gradient when the unit is not active:

      f(x) = alpha * x if x < 0
      f(x) = x if x >= 0
    Usage:

    >> > layer = tf.keras.layers.LeakyReLU()
    >> > output = layer([-3.0, -1.0, 0.0, 2.0])
    >> > list(output.numpy())
    [-0.9, -0.3, 0.0, 2.0]
    >> > layer = tf.keras.layers.LeakyReLU(alpha=0.1)
    >> > output = layer([-3.0, -1.0, 0.0, 2.0])
    >> > list(output.numpy())
    [-0.3, -0.1, 0.0, 2.0]
    '''


# ---------------------------PReLU----------------------------
def part_3():
    tf.random.set_seed(42)
    np.random.seed(42)

    model = keras.models.Sequential([
        keras.layers.Flatten(input_shape=[28, 28]),
        keras.layers.Dense(300, kernel_initializer="he_normal"),
        keras.layers.PReLU(),
        keras.layers.Dense(100, kernel_initializer="he_normal"),
        keras.layers.PReLU(),
        keras.layers.Dense(10, activation="softmax")
    ])
    # 稀疏分类交叉熵 sparse categorical crossentropy
    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer=keras.optimizers.SGD(lr=1e-3),
                  metrics=["accuracy"])
    history = model.fit(X_train, y_train, epochs=10,
                        validation_data=(X_valid, y_valid))
    '''
    Parametric Rectified Linear Unit.
    在PReLU中，负部分的斜率是从数据中学习的，而不是预定义的。
    It follows:

      f(x) = alpha * x for x < 0
      f(x) = x for x >= 0
    where alpha is a learned array with the same shape as x.
    '''

# -----------------------------ELU--------------------------------


def part_4():

    def elu(z, alpha=1):
        return np.where(z < 0, alpha * (np.exp(z) - 1), z)

    plt.plot(z, elu(z), "b-", linewidth=2)
    plt.plot([-5, 5], [0, 0], 'k-')
    plt.plot([-5, 5], [-1, -1], 'k--')
    plt.plot([0, 0], [-2.2, 3.2], 'k-')
    plt.grid(True)
    plt.title(r"ELU activation function ($\alpha=1$)", fontsize=14)
    plt.axis([-5, 5, -2.2, 3.2])
    plt.show()
    # 在网络层使用ELU
    keras.layers.Dense(10, activation="elu")


# --------------------------SELU---------------------------------------
# 默认情况下，SELU超参数（`scale`和`alpha`）的调整方式是，
# 每个神经元的平均输出保持接近0，标准偏差保持接近1
# （假设输入也用平均值0和标准偏差1标准化）。
# 使用此激活函数，即使是1000层深度的神经网络，
# 也会在所有层中保留大致的平均值0和标准偏差1，从而避免梯度爆炸/消失问题
# alpha和scale以平均值0和标准偏差1自标准化
# 显示SELU图像
def part_5():
    alpha_0_1 = -np.sqrt(2 / np.pi) / (erfc(1/np.sqrt(2)) * np.exp(1/2) - 1)
    scale_0_1 = (1 - erfc(1 / np.sqrt(2)) * np.sqrt(np.e)) * np.sqrt(2 * np.pi) * (2 * erfc(np.sqrt(2))*np.e **
                                                                                   2 + np.pi*erfc(1/np.sqrt(2))**2*np.e - 2*(2+np.pi)*erfc(1/np.sqrt(2))*np.sqrt(np.e)+np.pi+2)**(-1/2)

    # 1.6732632423543778
    # 1.0507009873554805
    # print(alpha_0_1)
    # print(scale_0_1)
    # SELU

    def selu(z, scale=scale_0_1, alpha=alpha_0_1):
        def elu(a, alpha=1):
            return np.where(a < 0, alpha * (np.exp(a) - 1), a)
        return scale * elu(z, alpha)

    plt.plot(z, selu(z), "b-", linewidth=2)
    plt.plot([-5, 5], [0, 0], 'k-')
    plt.plot([-5, 5], [-1.758, -1.758], 'k--')
    plt.plot([0, 0], [-2.2, 3.2], 'k-')
    plt.grid(True)
    plt.title("SELU activation function", fontsize=14)
    plt.axis([-5, 5, -2.2, 3.2])
    plt.show()


# 模型中使用SELU
def part_6():
    np.random.seed(42)
    tf.random.set_seed(42)
    model = keras.models.Sequential()
    model.add(keras.layers.Flatten(input_shape=[28, 28]))
    model.add(keras.layers.Dense(300, activation="selu",
                                 kernel_initializer="lecun_normal"))
    for layer in range(99):
        model.add(keras.layers.Dense(100, activation="selu",
                                     kernel_initializer="lecun_normal"))
    model.add(keras.layers.Dense(10, activation="softmax"))
    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer=keras.optimizers.SGD(lr=1e-3),
                  metrics=["accuracy"])
    pixel_means = X_train.mean(axis=0, keepdims=True)  # 求平均值
    pixel_stds = X_train.std(axis=0, keepdims=True)  # 标准差
    # 特征缩放
    X_train_scaled = (X_train - pixel_means) / pixel_stds
    X_valid_scaled = (X_valid - pixel_means) / pixel_stds
    X_test_scaled = (X_test - pixel_means) / pixel_stds
    history = model.fit(X_train_scaled, y_train, epochs=5,
                        validation_data=(X_valid_scaled, y_valid))

# 使用ReLu代替part6中的SELU
# 可能会遇见消失/爆炸梯度问题


def part_7():
    root_logdir = os.path.join(os.curdir, "my_logs")

    def get_run_logdir():
        import time
        run_id = time.strftime("run_%Y_%m_%d_%H_%M_%S")
        return os.path.join(root_logdir, run_id)

    run_logdir = get_run_logdir()
    tensorboard_cb = keras.callbacks.TensorBoard(
        log_dir=run_logdir, write_images=True, histogram_freq=1, write_grads=True)
    # $ python -m tensorboard.main --logdir=./my_logs --port=6060
    np.random.seed(42)
    tf.random.set_seed(42)
    model = keras.models.Sequential()
    model.add(keras.layers.Flatten(input_shape=[28, 28]))
    model.add(keras.layers.Dense(300, activation="relu",
                                 kernel_initializer="he_normal"))
    for layer in range(99):
        model.add(keras.layers.Dense(100, activation="relu",
                                     kernel_initializer="he_normal"))
    model.add(keras.layers.Dense(10, activation="softmax"))
    model.compile(loss="sparse_categorical_crossentropy",
                  optimizer=keras.optimizers.SGD(lr=1e-3),
                  metrics=["accuracy"])
    pixel_means = X_train.mean(axis=0, keepdims=True)  # 求平均值
    pixel_stds = X_train.std(axis=0, keepdims=True)  # 标准差
    X_train_scaled = (X_train - pixel_means) / pixel_stds
    X_valid_scaled = (X_valid - pixel_means) / pixel_stds
    X_test_scaled = (X_test - pixel_means) / pixel_stds
    history = model.fit(X_train_scaled, y_train, epochs=5,
                        validation_data=(X_valid_scaled, y_valid),
                        callbacks=[tensorboard_cb])


part_7()
