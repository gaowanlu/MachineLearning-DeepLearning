import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# mnist衣物数据准备
fashion_mnist = keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()

# 训练集与验证集
X_valid, X_train = X_train_full[:5000] / \
    255.0, X_train_full[5000:] / 255.0  # 归一化
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

# 标签
class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
               "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

# 显示前40个样本 4行10列
n_rows = 4
n_cols = 10
plt.figure(figsize=(n_cols * 1.2, n_rows * 1.2))
for row in range(n_rows):
    for col in range(n_cols):
        index = n_cols * row + col
        plt.subplot(n_rows, n_cols, index + 1)
        plt.imshow(X_train[index], cmap="binary", interpolation="nearest")
        plt.axis('off')
        plt.title(class_names[y_train[index]], fontsize=12)
plt.subplots_adjust(wspace=0.2, hspace=0.5)
plt.show()

# 使用顺序API创建模型
model = keras.models.Sequential()
# 输入数据格式 Flatten层 将每个输入图像转换为一维的数组
# 或者使用 keras.layers.InputLayer 作为第一层 input_shape=[28,28]
model.add(keras.layers.Flatten(input_shape=[28, 28]))
# 拥有300个神经元的Dense隐藏层使用ReLU激活函数
# 等效于 activation=keras.activations.relu
model.add(keras.layers.Dense(300, activation="relu"))
# 100个神经元的Dense隐藏层 ReLu 为激活函数
model.add(keras.layers.Dense(100, activation="relu"))
# 10个神经元的Dense输出层 每个类别一个 使用softmax为激活函数
model.add(keras.layers.Dense(10, activation="softmax"))

# 或者使用构造函数构建model
# model=keras.models.Sequential([
#     keras.layers.Flatten(input_shape=[28, 28]),
#     keras.layers.Dense(300, activation="relu"),
#     keras.layers.Dense(100, activation="relu"),
#     keras.layers.Dense(10, activation="softmax")
# ])

print(model.layers)
print(model.output_shape)  # (None, 10)
print(model.summary())
'''
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
flatten (Flatten)            (None, 784)               0
_________________________________________________________________
dense (Dense)                (None, 300)               235500 
第一个隐藏层的连接权重为784*300 ，外加300个偏置项共有235500个参数
_________________________________________________________________
dense_1 (Dense)              (None, 100)               30100
_________________________________________________________________
dense_2 (Dense)              (None, 10)                1010
=================================================================
Total params: 266,610
Trainable params: 266,610
Non-trainable params: 0
_________________________________________________________________
None
'''
# 画出模型
keras.utils.plot_model(model, "my_fashion_mnist_model.png", show_shapes=True)

# 可以使用get_weights 与 set_weights访问层的所有参数
weights, biases = (model.layers[1].get_weights())
print("权重", weights)
print("偏置", biases)

# 编译模型
# 损失 优化器 指标
#optimizer = keras.optimizers.SGD(learning_rate=0.5)
model.compile(loss=keras.losses.sparse_categorical_crossentropy,
              optimizer=keras.optimizers.SGD(),
              metrics=[keras.metrics.sparse_categorical_accuracy])
print("编译模型完成")

history = model.fit(X_train, y_train, epochs=30,
                    validation_data=(X_valid, y_valid))
# 返回的history对象 包含训练参数 history.params \经历的轮次列表 history.epoch
# 在训练集验证集上的每个轮次结束时测得的损失和额外指标字典 history.history
pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.show()

# 使用模型进行预测
X_new = X_test[:3]
y_proba = model.predict(X_new)
print(y_proba.round(2))  # round(2) list 保留两位小数
# [[0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]
#  [0. 0. 1. 0. 0. 0. 0. 0. 0. 0.]
#  [0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]]
# 每行的和为1 总概率为1
y_pred = model.predict_classes(X_new)
print(y_pred)
print(y_test[:3])
print(np.array(class_names)[y_pred])
# [9 2 1] 全部预测正确
# [9 2 1]
# ['Ankle boot' 'Pullover' 'Trouser']
