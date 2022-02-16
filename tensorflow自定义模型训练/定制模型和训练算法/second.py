from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow.keras as keras
import numpy as np
import time
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
input_shape = X_train.shape[1:]

# 使用工厂模式对损失函数进行封装--------------------


def create_huber(threshold=1.0):
    def huber_fn(y_true, y_pred):
        error = y_true - y_pred
        is_small_error = tf.abs(error) < threshold
        squared_loss = tf.square(error) / 2
        linear_loss = threshold * tf.abs(error) - threshold**2 / 2
        return tf.where(is_small_error, squared_loss, linear_loss)
    return huber_fn


keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)
# 模型构建
model = keras.models.Sequential([
    keras.layers.Dense(30, activation="selu", kernel_initializer="lecun_normal",
                       input_shape=input_shape),
    keras.layers.Dense(1),
])
# 编译
model.compile(loss="mse", optimizer="nadam", metrics=[create_huber(2.0)])
# 训练
model.fit(X_train_scaled, y_train, epochs=2)


model.compile(loss=create_huber(2.0), optimizer="nadam",
              metrics=[create_huber(2.0)])
sample_weight = np.random.rand(len(y_train))
history = model.fit(X_train_scaled, y_train, epochs=2,
                    sample_weight=sample_weight)
print(history.history["loss"][0])  # 0.11749907582998276
print(history.history["huber_fn"][0] *
      sample_weight.mean())  # 0.11906625573138947

# keras.metrics.Precision 精准率
precision = keras.metrics.Precision()
precision.update_state([0, 1, 1, 1, 0, 1, 0, 1], [1, 1, 0, 1, 0, 1, 0, 1])
print(precision.result())  # tf.Tensor(0.8, shape=(), dtype=float32) 6/8=0.75
print(precision.variables)
'''
[<tf.Variable 'true_positives:0' shape=(1,) dtype=float32, numpy=array([4.], dtype=float32)>,
    <tf.Variable 'false_positives:0' shape=(1,) dtype=float32, numpy=array([1.], dtype=float32)>]
'''
print(precision.reset_states())  # None


# Creating a streaming metric 自定义Metric--------------------
class HuberMetric(keras.metrics.Metric):
    def __init__(self, threshold=1.0, **kwargs):
        super().__init__(**kwargs)  # handles base args (e.g., dtype)
        self.threshold = threshold
        # self.huber_fn = create_huber(threshold) # TODO: investigate why this fails
        self.total = self.add_weight("total", initializer="zeros")
        self.count = self.add_weight("count", initializer="zeros")

    def huber_fn(self, y_true, y_pred):  # workaround
        error = y_true - y_pred
        is_small_error = tf.abs(error) < self.threshold
        squared_loss = tf.square(error) / 2
        linear_loss = self.threshold * tf.abs(error) - self.threshold**2 / 2
        return tf.where(is_small_error, squared_loss, linear_loss)

    def update_state(self, y_true, y_pred, sample_weight=None):
        metric = self.huber_fn(y_true, y_pred)
        self.total.assign_add(tf.reduce_sum(metric))
        self.count.assign_add(tf.cast(tf.size(y_true), tf.float32))

    def result(self):
        return self.total / self.count

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "threshold": self.threshold}


m = HuberMetric(2.)

# total = 2 * |10 - 2| - 2²/2 = 14
# count = 1
# result = 14 / 1 = 14
m.update_state(tf.constant([[2.]]), tf.constant([[10.]]))
# total = total + (|1 - 0|² / 2) + (2 * |9.25 - 5| - 2² / 2) = 14 + 7 = 21
# count = count + 2 = 3
# result = total / count = 21 / 3 = 7
m.update_state(tf.constant([[0.], [5.]]), tf.constant([[1.], [9.25]]))

print(m.result())
print(m.variables)
print(m.reset_states())
print(m.variables)
'''
tf.Tensor(7.0, shape=(), dtype=float32)
[<tf.Variable 'total:0' shape=() dtype=float32, numpy=21.0>, <tf.Variable 'count:0' shape=() dtype=float32, numpy=3.0>]
None
[<tf.Variable 'total:0' shape=() dtype=float32, numpy=0.0>, <tf.Variable 'count:0' shape=() dtype=float32, numpy=0.0>]
'''

# 使用自定义的Metric构建模型
keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)
# 构建
model = keras.models.Sequential([
    keras.layers.Dense(30, activation="selu", kernel_initializer="lecun_normal",
                       input_shape=input_shape),
    keras.layers.Dense(1),
])
# 编译
model.compile(loss=create_huber(2.0), optimizer="nadam",
              metrics=[HuberMetric(2.0)])
# 训练
model.fit(X_train_scaled.astype(np.float32),
          y_train.astype(np.float32), epochs=2)
# 保存模型
model.save("my_model_with_a_custom_metric.h5")
# model = keras.models.load_model("my_model_with_a_custom_metric.h5",
#                                custom_objects={"huber_fn": create_huber(2.0),
#                                                "HuberMetric": HuberMetric})
model.fit(X_train_scaled.astype(np.float32),
          y_train.astype(np.float32), epochs=2)
print(model.metrics[-1].threshold)  # 2.0


# 继承现有的metrics 实现自定义Metrics
class HuberMetric(keras.metrics.Mean):
    def __init__(self, threshold=1.0, name='HuberMetric', dtype=None):
        self.threshold = threshold
        self.huber_fn = create_huber(threshold)
        super().__init__(name=name, dtype=dtype)

    def update_state(self, y_true, y_pred, sample_weight=None):
        metric = self.huber_fn(y_true, y_pred)
        super(HuberMetric, self).update_state(metric, sample_weight)

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "threshold": self.threshold}


keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)
model = keras.models.Sequential([
    keras.layers.Dense(30, activation="selu", kernel_initializer="lecun_normal",
                       input_shape=input_shape),
    keras.layers.Dense(1),
])
model.compile(loss=keras.losses.Huber(2.0), optimizer="nadam",
              weighted_metrics=[HuberMetric(2.0)])
sample_weight = np.random.rand(len(y_train))
history = model.fit(X_train_scaled.astype(np.float32), y_train.astype(np.float32),
                    epochs=2, sample_weight=sample_weight)
print(history.history["loss"][0])
print(history.history["HuberMetric"][0] * sample_weight.mean())
model.save("my_model_with_a_custom_metric_v2.h5")
# model = keras.models.load_model("my_model_with_a_custom_metric_v2.h5",        # TODO: check PR #25956
#                                custom_objects={"HuberMetric": HuberMetric})
model.fit(X_train_scaled.astype(np.float32),
          y_train.astype(np.float32), epochs=2)
print(model.metrics[-1].threshold)


# Custom Layers 自定义网络层
exponential_layer = keras.layers.Lambda(lambda x: tf.exp(x))
exponential_layer([-1., 0., 1.])

# 如果要预测的值为正值且具有非常不同的标度，则在回归模型的输出处添加指数层是有用的(e.g., 0.001, 10., 10000):

keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)
model = keras.models.Sequential([
    keras.layers.Dense(30, activation="relu", input_shape=input_shape),
    keras.layers.Dense(1),
    exponential_layer
])
model.compile(loss="mse", optimizer="nadam")
model.fit(X_train_scaled, y_train, epochs=5,
          validation_data=(X_valid_scaled, y_valid))
print(model.evaluate(X_test_scaled, y_test))
# None
# layers.Lambda不是Layer
# 接着看

# 使用继承 自定义网络层


class MyDense(keras.layers.Layer):
    def __init__(self, units, activation=None, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.activation = keras.activations.get(activation)

    def build(self, batch_input_shape):
        self.kernel = self.add_weight(
            name="kernel", shape=[batch_input_shape[-1], self.units],
            initializer="glorot_normal")
        self.bias = self.add_weight(
            name="bias", shape=[self.units], initializer="zeros")
        super().build(batch_input_shape)  # must be at the end

    def call(self, X):
        return self.activation(X @ self.kernel + self.bias)

    def compute_output_shape(self, batch_input_shape):
        return tf.TensorShape(batch_input_shape.as_list()[:-1] + [self.units])

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "units": self.units,
                "activation": keras.activations.serialize(self.activation)}


keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)
model = keras.models.Sequential([
    MyDense(30, activation="relu", input_shape=input_shape),
    MyDense(1)
])
model.compile(loss="mse", optimizer="nadam")
model.fit(X_train_scaled, y_train, epochs=2,
          validation_data=(X_valid_scaled, y_valid))
model.evaluate(X_test_scaled, y_test)
model.save("my_model_with_a_custom_layer.h5")
model = keras.models.load_model("my_model_with_a_custom_layer.h5",
                                custom_objects={"MyDense": MyDense})


# 自定义多输入网络层
class MyMultiLayer(keras.layers.Layer):
    def call(self, X):
        X1, X2 = X
        return X1 + X2, X1 * X2

    def compute_output_shape(self, batch_input_shape):  # 多输出
        batch_input_shape1, batch_input_shape2 = batch_input_shape
        return [batch_input_shape1, batch_input_shape2]


keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)
inputs1 = keras.layers.Input(shape=[2])
inputs2 = keras.layers.Input(shape=[2])
outputs1, outputs2 = MyMultiLayer()((inputs1, inputs2))
print("outputs1", outputs1)
print("outputs2", outputs2)
'''
outputs1 KerasTensor(type_spec=TensorSpec(shape=(None, 2), dtype=tf.float32, name=None), name='my_multi_layer/add:0', description="created by layer 'my_multi_layer'")
outputs2 KerasTensor(type_spec=TensorSpec(shape=(None, 2), dtype=tf.float32, name=None), name='my_multi_layer/mul:0', description="created by layer 'my_multi_layer'")
'''

class AddGaussianNoise(keras.layers.Layer):
    def __init__(self, stddev, **kwargs):
        super().__init__(**kwargs)
        self.stddev = stddev

    def call(self, X, training=None):
        if training:
            noise = tf.random.normal(tf.shape(X), stddev=self.stddev)
            return X + noise
        else:
            return X

    def compute_output_shape(self, batch_input_shape):
        return batch_input_shape


model.compile(loss="mse", optimizer="nadam")
model.fit(X_train_scaled, y_train, epochs=2,
          validation_data=(X_valid_scaled, y_valid))
print(model.evaluate(X_test_scaled, y_test))  # 0.3990039527416229
