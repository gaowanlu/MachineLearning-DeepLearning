#树的数量 提前停止  
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
import matplotlib.pyplot as plt


#画出决策树回归的图
def plot_predictions(regressors, X, y, axes, label=None, style="r-", data_style="b.", data_label=None):
    x1 = np.linspace(axes[0], axes[1], 500)
    y_pred = sum(regressor.predict(x1.reshape(-1, 1)) for regressor in regressors)
    plt.plot(X[:, 0], y, data_style, label=data_label)
    plt.plot(x1, y_pred, style, linewidth=2, label=label)
    if label or data_label:
        plt.legend(loc="upper center", fontsize=16)
    plt.axis(axes)


#数据准备
np.random.seed(42)
X = np.random.rand(100, 1) - 0.5
y = 3*X[:, 0]**2 + 0.05 * np.random.randn(100)
X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=49)
#梯度提升法
gbrt = GradientBoostingRegressor(max_depth=2, n_estimators=120, random_state=42)
gbrt.fit(X_train, y_train)

#平方均差 测试树的数量0-120 的每个情况的损失 找到损失最小的情况
#暴力枚举，并不是提前停止，提前停止在最下面
errors = [mean_squared_error(y_val, y_pred) for y_pred in gbrt.staged_predict(X_val)]
bst_n_estimators = np.argmin(errors) + 1

#利用损失最小的树的数量训练模型
gbrt_best = GradientBoostingRegressor(max_depth=2, n_estimators=bst_n_estimators, random_state=42)
gbrt_best.fit(X_train, y_train)


# 画图
min_error = np.min(errors)
plt.figure(figsize=(10, 4))
plt.subplot(121)
plt.plot(errors, "b.-")
plt.plot([bst_n_estimators, bst_n_estimators], [0, min_error], "k--")
plt.plot([0, 120], [min_error, min_error], "k--")

plt.plot(bst_n_estimators, min_error, "ko")
plt.text(bst_n_estimators, min_error*1.2, "Minimum", ha="center", fontsize=14)

plt.axis([0, 120, 0, 0.01])
plt.xlabel("Number of trees")
plt.ylabel("Error", fontsize=16)
plt.title("Validation error", fontsize=14)

plt.subplot(122)
plot_predictions([gbrt_best], X, y, axes=[-0.5, 0.5, -0.1, 0.8])
plt.title("Best model (%d trees)" % bst_n_estimators, fontsize=14)
plt.ylabel("$y$", fontsize=16, rotation=0)
plt.xlabel("$x_1$", fontsize=16)

plt.show()




# 提前停止法
# 要实现提前停止法，不一定需要先训练大量的树，然后再回头找最优的数
# 字，还可以提前停止训练。设置warm_start=True，当fit（）方法被调用时，Scikit-Learn会
# 保留现有的树，从而允许增量训练。以下代码会在验证误差连续5次迭代未改善时，直接
# 停止训练
gbrt = GradientBoostingRegressor(max_depth=2, warm_start=True, random_state=42)
min_val_error = float("inf")
error_going_up = 0
for n_estimators in range(1, 120):#树数量从1到120迭代
    gbrt.n_estimators = n_estimators#设置树数量
    gbrt.fit(X_train, y_train)#训练
    y_pred = gbrt.predict(X_val)
    val_error = mean_squared_error(y_val, y_pred)#计算误差
    if val_error < min_val_error:#误差更小则迭代更优
        min_val_error = val_error
        error_going_up = 0
    else:
        error_going_up += 1
        if error_going_up == 5:#连续5次误差不下降则提前停止
            break  # 停止
print(gbrt.n_estimators)#61

'''
随机梯度提升
GradientBoostingRegressor类还可以支持超参数subsample，指定用于训练每棵树的实
例的比例。例如，如果subsample=0.25，则每棵树用25%的随机选择的实例进行训练。
用更高的偏差换取了更低的方差，同时在相当大的程度上加速了训练过程。

梯度提升也可以使用其他成本函数，通过超参数loss来控制

'''