from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import reciprocal
from tensorflow import keras
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing

keras.backend.clear_session()
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
# -------------------------------------------------------------------


def build_model(n_hidden=1, n_neurons=30, learning_rate=3e-3, input_shape=[8]):
    model = keras.models.Sequential()
    # 输入层
    model.add(keras.layers.InputLayer(input_shape=input_shape))
    # 隐藏层
    for layer in range(n_hidden):
        # 神经元个数
        model.add(keras.layers.Dense(n_neurons, activation="relu"))
    # 输出层
    model.add(keras.layers.Dense(1))
    optimizer = keras.optimizers.SGD(lr=learning_rate)  #
    # 编译模型
    model.compile(loss="mse", optimizer=optimizer)
    return model


keras_reg = keras.wrappers.scikit_learn.KerasRegressor(build_model)
# 数据训练
# keras_reg.fit(X_train, y_train, epochs=100,
#               validation_data=(X_valid, y_valid),
#               callbacks=[keras.callbacks.EarlyStopping(patience=10)])
# # 模型评估
# mse_test = keras_reg.score(X_test, y_test)
# # 预测
# y_pred = keras_reg.predict(X_test[:3])

print("参数寻找")

print("学习率", reciprocal(3e-4, 3e-2))

# 超参数调整 类似于 传统机器学习的grid search
param_distribs = {
    "n_hidden": [0, 1, 2, 3],  # 隐藏层数
    "n_neurons": np.arange(1, 10),  # 每层神经元个数
    "learning_rate": reciprocal(3e-4, 3e-2),  # 学习率
}
# 随即搜索
rnd_search_cv = RandomizedSearchCV(
    keras_reg, param_distribs, n_iter=10, cv=3, verbose=2)

earlyStop = keras.callbacks.EarlyStopping(patience=10)

rnd_search_cv.fit(X_train, y_train, epochs=10, validation_data=(
    X_valid, y_valid), callbacks=[earlyStop])  #

print("rnd_search_cv.best_params_", rnd_search_cv.best_params_)
print("rnd_search_cv.best_score_", rnd_search_cv.best_score_)
print("rnd_search_cv.best_estimator_", rnd_search_cv.best_estimator_)

test_score = rnd_search_cv.score(X_test, y_test)
print("test_score", test_score)
model = rnd_search_cv.best_estimator_.model
print("best model evaluate", model.evaluate(X_test, y_test))
# BUG BUG 没解决
'''
Traceback(most recent call last):
File "ParamsAdjust.py", line 70, in <module >
rnd_search_cv.fit(X_train, y_train, epochs=10, validation_data=(
File "D:\anaconda\envs\tensorflow\lib\site-packages\sklearn\model_selection\_search.py", line 921, in fit
self.best_estimator_=clone(
File "D:\anaconda\envs\tensorflow\lib\site-packages\sklearn\base.py", line 95, in clone
raise RuntimeError(
RuntimeError: Cannot clone object < tensorflow.python.keras.wrappers.scikit_learn.KerasRegressor 
object at 0x0000021F971FE100 > , as the constructor either does not set or modifies parameter n_neurons
'''
