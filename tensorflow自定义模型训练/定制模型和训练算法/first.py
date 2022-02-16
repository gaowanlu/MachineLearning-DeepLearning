# Custom loss function 自定义损失函数
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow.keras as keras
import numpy as np

'''
加载加州住房数据集，获得训练集 验证集 测试集 让后对他们进行缩放处理
'''
# 数据加载 数据集划分
housing = fetch_california_housing()
X_train_full, X_test, y_train_full, y_test = train_test_split(
    housing.data, housing.target.reshape(-1, 1), random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(
    X_train_full, y_train_full, random_state=42)
# 缩放处理
scaler = StandardScaler()
# .fit_transform不仅计算训练数据的均值和方差，还会基于计算出来的均值和方差来转换训练数据，从而把数据转换成标准的正太分布
X_train_scaled = scaler.fit_transform(X_train)
# .transform它只是进行转换，只是把训练数据转换成标准的正态分布
X_valid_scaled = scaler.transform(X_valid)
X_test_scaled = scaler.transform(X_test)
# .fit 用于计算训练数据的均值和方差， 后面就会用均值和方差来转换训练数据

# 定义损失函数--------------------
# 在 tf.losses.Huber 中有huber损失函数 为了学习进行自定义模拟


def huber_fn(y_true, y_pred):
    error = y_true - y_pred  # y-y^
    is_small_error = tf.abs(error) < 1
    squared_loss = tf.square(error) / 2
    linear_loss = tf.abs(error) - 0.5
    return tf.where(is_small_error, squared_loss, linear_loss)

# 画出huber_fn函数图像


def draw_huber():
    plt.figure(figsize=(8, 3.5))
    z = np.linspace(-4, 4, 200)
    plt.plot(z, huber_fn(0, z), "b-", linewidth=2, label="huber($z$)")
    plt.plot(z, z**2 / 2, "b:", linewidth=1, label=r"$\frac{1}{2}z^2$")
    plt.plot([-1, -1], [0, huber_fn(0., -1.)], "r--")
    plt.plot([1, 1], [0, huber_fn(0., 1.)], "r--")
    plt.gca().axhline(y=0, color='k')
    plt.gca().axvline(x=0, color='k')
    plt.axis([-4, 4, 0, 4])
    plt.grid(True)
    plt.xlabel("$z$")
    plt.legend(fontsize=14)
    plt.title("Huber loss", fontsize=14)
    plt.show()


draw_huber()


input_shape = X_train.shape[1:]
# 构建神经网络
model = keras.models.Sequential([
    keras.layers.Dense(30, activation="selu", kernel_initializer="lecun_normal",
                       input_shape=input_shape),
    keras.layers.Dense(1),
])
# 编译
model.compile(loss=huber_fn, optimizer="nadam", metrics=["mae"])  # 使用自定义损失函数
# 训练
history = model.fit(X_train_scaled, y_train, epochs=2,
                    validation_data=(X_valid_scaled, y_valid))
print(model.summary())

# 保存模型
model.save("my_model_with_a_custom_loss.h5")
# 加载模型 模型中并没有保存自定定义的损失函数 huber_fn 则需要使用custom对象进行key:value映射赋值
# 否则模型中找不到huber_fn 则会报错
model = keras.models.load_model("my_model_with_a_custom_loss.h5",
                                custom_objects={"huber_fn": huber_fn})
model.fit(X_train_scaled, y_train, epochs=2,
          validation_data=(X_valid_scaled, y_valid))


# 使用工厂模式对损失函数进行封装--------------------
def create_huber(threshold=1.0):
    def huber_fn(y_true, y_pred):
        error = y_true - y_pred
        is_small_error = tf.abs(error) < threshold
        squared_loss = tf.square(error) / 2
        linear_loss = threshold * tf.abs(error) - threshold**2 / 2
        return tf.where(is_small_error, squared_loss, linear_loss)
    return huber_fn


# 编译
model.compile(loss=create_huber(2.0), optimizer="nadam", metrics=["mae"])
# 训练
model.fit(X_train_scaled, y_train, epochs=2,
          validation_data=(X_valid_scaled, y_valid))
# 保存
model.save("my_model_with_a_custom_loss_threshold_2.h5")
# 加载模型
model = keras.models.load_model("my_model_with_a_custom_loss_threshold_2.h5",
                                custom_objects={"huber_fn": create_huber(2.0)})
# 训练
model.fit(X_train_scaled, y_train, epochs=2,
          validation_data=(X_valid_scaled, y_valid))


# 类继承keras.losses.Loss 自定义损失函数--------------------
class HuberLoss(keras.losses.Loss):
    def __init__(self, threshold=1.0, **kwargs):
        self.threshold = threshold
        super().__init__(**kwargs)

    def call(self, y_true, y_pred):
        error = y_true - y_pred
        is_small_error = tf.abs(error) < self.threshold
        squared_loss = tf.square(error) / 2
        linear_loss = self.threshold * tf.abs(error) - self.threshold**2 / 2
        return tf.where(is_small_error, squared_loss, linear_loss)

    def get_config(self):
        base_config = super().get_config()
        # **base_config对象解构
        return {**base_config, "threshold": self.threshold}


# 构建网络模型
model = keras.models.Sequential([
    keras.layers.Dense(30, activation="selu", kernel_initializer="lecun_normal",
                       input_shape=input_shape),
    keras.layers.Dense(1),
])
# 编译 阈值2.0
model.compile(loss=HuberLoss(2.), optimizer="nadam", metrics=["mae"])
# 训练
model.fit(X_train_scaled, y_train, epochs=2,
          validation_data=(X_valid_scaled, y_valid))
model.save("my_model_with_a_custom_loss_class.h5")
# 使用custom_objects 赋值 HuberLoss
model = keras.models.load_model("my_model_with_a_custom_loss_class.h5",
                                custom_objects={"HuberLoss": HuberLoss})
model.fit(X_train_scaled, y_train, epochs=2,
          validation_data=(X_valid_scaled, y_valid))
print(model.loss.threshold)  # 2.0


# 其他自定义函数--------------------
keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)

# 自定义激活函数 keras.activations.softplus


def my_softplus(z):  # return value is just tf.nn.softplus(z)
    return tf.math.log(tf.exp(z) + 1.0)

# 自定义Glorot初始化


def my_glorot_initializer(shape, dtype=tf.float32):
    stddev = tf.sqrt(2. / (shape[0] + shape[1]))
    return tf.random.normal(shape, stddev=stddev, dtype=dtype)

# 自定义1正则化


def my_l1_regularizer(weights):
    return tf.reduce_sum(tf.abs(0.01 * weights))

# 确保权重均为正的自定义约束


def my_positive_weights(weights):  # return value is just tf.nn.relu(weights)
    return tf.where(weights < 0., tf.zeros_like(weights), weights)


# 自定义网络层
layer = keras.layers.Dense(1, activation=my_softplus,
                           kernel_initializer=my_glorot_initializer,
                           kernel_regularizer=my_l1_regularizer,
                           kernel_constraint=my_positive_weights)
# 构建网络模型
model = keras.models.Sequential([
    keras.layers.Dense(30, activation="selu", kernel_initializer="lecun_normal",
                       input_shape=input_shape),
    keras.layers.Dense(1, activation=my_softplus,
                       kernel_regularizer=my_l1_regularizer,
                       kernel_constraint=my_positive_weights,
                       kernel_initializer=my_glorot_initializer),
])
# 编译
model.compile(loss="mse", optimizer="nadam", metrics=["mae"])
# 训练
model.fit(X_train_scaled, y_train, epochs=2,
          validation_data=(X_valid_scaled, y_valid))
# 保存模型
model.save("my_model_with_many_custom_parts.h5")
# 加载模型
model = keras.models.load_model(
    "my_model_with_many_custom_parts.h5",
    custom_objects={
        "my_l1_regularizer": my_l1_regularizer,
        "my_positive_weights": my_positive_weights,
        "my_glorot_initializer": my_glorot_initializer,
        "my_softplus": my_softplus,
    })


# 类封装1正则化
class MyL1Regularizer(keras.regularizers.Regularizer):
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, weights):
        return tf.reduce_sum(tf.abs(self.factor * weights))

    def get_config(self):
        return {"factor": self.factor}


keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)
# 构建网络模型
model = keras.models.Sequential([
    keras.layers.Dense(30, activation="selu", kernel_initializer="lecun_normal",
                       input_shape=input_shape),
    keras.layers.Dense(1, activation=my_softplus,
                       kernel_regularizer=MyL1Regularizer(0.01),  # 自定义正则化函数
                       kernel_constraint=my_positive_weights,
                       kernel_initializer=my_glorot_initializer),
])
# 编译
model.compile(loss="mse", optimizer="nadam", metrics=["mae"])
# 模型保存
model.save("my_model_with_many_custom_parts.h5")
# 模型加载
model = keras.models.load_model(
    "my_model_with_many_custom_parts.h5",
    custom_objects={
        "MyL1Regularizer": MyL1Regularizer,
        "my_positive_weights": my_positive_weights,
        "my_glorot_initializer": my_glorot_initializer,
        "my_softplus": my_softplus,
    })
# 训练
model.fit(X_train_scaled, y_train, epochs=2,
          validation_data=(X_valid_scaled, y_valid))
