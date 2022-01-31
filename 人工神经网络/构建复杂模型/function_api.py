import os
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing
from tensorflow import keras
import tensorflow as tf
import numpy as np
# 有些复杂的模型并不是层层递进的 可能输入层还与第一个隐藏层之外的层进行连接
np.random.seed(42)
tf.random.set_seed(42)
# 数据准备
housing = fetch_california_housing()

X_train_full, X_test, y_train_full, y_test = train_test_split(
    housing.data, housing.target, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(
    X_train_full, y_train_full, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)

# ---------------------------------------------------------------------

# 使用函数API
# 输入层
input_ = keras.layers.Input(shape=X_train.shape[1:])
# 隐藏层1 连接输入层
hidden1 = keras.layers.Dense(30, activation="relu")(input_)
# 隐藏层2 连接隐藏层1
hidden2 = keras.layers.Dense(30, activation="relu")(hidden1)
# 合并层 隐藏层2与输入层合并
concat = keras.layers.concatenate([input_, hidden2])
# 输出层 连接合并层
output = keras.layers.Dense(1)(concat)
# 构建模型
model = keras.models.Model(inputs=[input_], outputs=[output])
# 模型编译
model.compile(loss="mean_squared_error",
              optimizer=keras.optimizers.SGD(lr=1e-3))
# 数据训练
history = model.fit(X_train, y_train, epochs=20,
                    validation_data=(X_valid, y_valid))
# 模型评估
mse_test = model.evaluate(X_test, y_test)
# 模型预测
y_pred = model.predict(X_test[:3])
print(y_pred)
# [[0.4701073][1.8735044][3.379823]]
keras.utils.plot_model(model, "model_1.png", show_shapes=True)

# ---------------------------------------------------------------------
# fit回调函数
checkpoint_cb = keras.callbacks.ModelCheckpoint(
    "my_keras_model.h5", save_best_only=True)  # 模型fit完成时保存模型
# 提前停止 patience次轮回都没进展则提前停止
early_stopping_cb = keras.callbacks.EarlyStopping(patience=10,
                                                  restore_best_weights=True)
# 多输入多输出
# 两个输入层 一个输入5个特征 一个输入6个特征
input_A = keras.layers.Input(shape=[5], name="wide_input")
input_B = keras.layers.Input(shape=[6], name="deep_input")
# 两个隐藏层
hidden1 = keras.layers.Dense(30, activation="relu")(input_B)
hidden2 = keras.layers.Dense(30, activation="relu")(hidden1)
# 合并隐藏2与输入A
concat = keras.layers.concatenate([input_A, hidden2])
# 两个输出层
output = keras.layers.Dense(1, name="main_output")(concat)
aux_output = keras.layers.Dense(1, name="aux_output")(hidden2)
model = keras.models.Model(
    inputs=[input_A, input_B], outputs=[output, aux_output])
# 模型编译 指定输出损失的权重
model.compile(loss=["mse", "mse"], loss_weights=[0.9, 0.1],
              optimizer=keras.optimizers.SGD(lr=1e-3))

X_train_A, X_train_B = X_train[:, :5], X_train[:, 2:]
X_valid_A, X_valid_B = X_valid[:, :5], X_valid[:, 2:]
X_test_A, X_test_B = X_test[:, :5], X_test[:, 2:]
X_new_A, X_new_B = X_test_A[:3], X_test_B[:3]
# 数据训练
history = model.fit((X_train_A, X_train_B), [y_train, y_train], epochs=20,
                    validation_data=([X_valid_A, X_valid_B],
                                     [y_valid, y_valid]),
                    callbacks=[checkpoint_cb, early_stopping_cb])
# 模型评估
mse_test = model.evaluate((X_test_A, X_test_B), [y_test, y_test])
y_pred = model.predict((X_new_A, X_new_B))
keras.utils.plot_model(model, "model_2.png", show_shapes=True)

# ---------------------------------------------------------------------
# 以上为函数式API风格 另外还有子类API 自定义class继承keras.Model 可以查找资料查看

# # fit后将模型的保存
# model.save("my_keras_model.h5")
# # 模型加载
# model = keras.models.load_model("my_keras_model.h5")


# 还可以自定义回调函数 像切面编程 有许多生命周期钩子可供使用
# class PrintValTrainRatioCallback(keras.callbacks.Callback):
#     def on_epoch_end(self, epoch, logs):
#         print("\nval/train: {:.2f}".format(logs["val_loss"] / logs["loss"]))


# val_train_ratio_cb = PrintValTrainRatioCallback()
# history = model.fit(X_train, y_train, epochs=1,
#                     validation_data=(X_valid, y_valid),
#                     callbacks=[val_train_ratio_cb])


# -----------------------------------------------------------------

# 可视化工具tensorflow提供了TensorBoard
# $python tensorboard
root_logdir = os.path.join(os.curdir, "my_logs")


def get_run_logdir():
    import time
    run_id = time.strftime("run_%Y_%m_%d_%H_%M_%S")
    return os.path.join(root_logdir, run_id)


run_logdir = get_run_logdir()
tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)
input_ = keras.layers.Input(shape=X_train.shape[1:])
# 隐藏层1 连接输入层
hidden1 = keras.layers.Dense(30, activation="relu")(input_)
# 隐藏层2 连接隐藏层1
hidden2 = keras.layers.Dense(30, activation="relu")(hidden1)
# 合并层 隐藏层2与输入层合并
concat = keras.layers.concatenate([input_, hidden2])
# 输出层 连接合并层
output = keras.layers.Dense(1)(concat)
# 构建模型
model = keras.models.Model(inputs=[input_], outputs=[output])
# 模型编译
model.compile(loss="mean_squared_error",
              optimizer=keras.optimizers.SGD(lr=1e-3))
# 数据训练
history = model.fit(X_train, y_train, epochs=20,
                    validation_data=(X_valid, y_valid),
                    callbacks=[tensorboard_cb])
# 模型评估
mse_test = model.evaluate(X_test, y_test)
# 模型预测
y_pred = model.predict(X_test[:3])
print(y_pred)
# [[0.4701073][1.8735044][3.379823]]
keras.utils.plot_model(model, "model_1.png", show_shapes=True)


# $ python -m tensorboard.main --logdir=./my_logs --port=6060
