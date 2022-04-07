import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# 数据准备 加载数据集
WINE_DATA_PATH = "./wine_data.csv"


def load_data(path):
    return pd.read_csv(path)


wine_data = load_data(WINE_DATA_PATH)


def data_format():
    X = []
    Y = []
    for i in range(len(wine_data)):
        item = []
        for j in range(len(wine_data.iloc[i])):
            if j == len(wine_data.iloc[i])-1:
                Y.append([wine_data.iloc[i][j]])
            else:
                item.append(wine_data.iloc[i][j])
        X.append(item)
    return (X, Y)


datas, label = data_format()

# 留出一部分作为测试集
datas_train = datas[:150]
labels_train = label[:150]
datas_test = datas[150:]
labels_test = label[150:]
print("datas_train {}".format(len(datas_train)))
print("datas_test {}".format(len(datas_test)))

# 数据集划分
X_train, X_valid, Y_train, Y_valid = train_test_split(
    datas_train, labels_train, random_state=42)

print("X_train {} Y_train {} X_valid {} Y_valid {}".format(
    len(X_train), len(Y_train), len(X_valid), len(Y_valid)))

# 标准化
scaler = StandardScaler()
X_train = np.array(scaler.fit_transform(X_train))
X_valid = np.array(scaler.fit_transform(X_valid))
X_test = np.array(scaler.fit_transform(datas_test))
Y_train = np.array(Y_train)
Y_valid = np.array(Y_valid)
Y_test = np.array(labels_test)


# 使用构造函数构建model
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[len(X_train[0])]),
    keras.layers.Dense(16, activation="relu"),
    keras.layers.Dense(10, activation="relu"),
    keras.layers.Dense(3, activation="softmax")
])

print(model.layers)
print(model.output_shape)  # (None, 10)
print(model.summary())


# 编译模型
# 损失 优化器 指标
#optimizer = keras.optimizers.SGD(learning_rate=0.5)
model.compile(loss=keras.losses.sparse_categorical_crossentropy,
              optimizer=keras.optimizers.SGD(),
              metrics=[keras.metrics.sparse_categorical_accuracy])
print("编译模型完成")

# 训练网络模型
history = model.fit(X_train, Y_train, epochs=1000,
                    validation_data=(X_valid, Y_valid))
# # 返回的history对象 包含训练参数 history.params \经历的轮次列表 history.epoch
# # 在训练集验证集上的每个轮次结束时测得的损失和额外指标字典 history.history
pd.DataFrame(history.history).plot(figsize=(10, 10))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.show()

# 使用模型进行预测
y_proba = model.predict(X_test)
print(y_proba.round(2))  # round(2) list 保留两位小数
# 每行的和为1 总概率为1

y_pred = model.predict_classes(X_test)
print(np.array([0, 1, 2])[y_pred])
print(Y_test)
