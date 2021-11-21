'''
XGBoost提供梯度提升的优化实现
'''
import xgboost
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


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


#没有早停止
xgb_reg = xgboost.XGBRegressor(random_state=42)
xgb_reg.fit(X_train, y_train)
y_pred = xgb_reg.predict(X_val)
val_error = mean_squared_error(y_val, y_pred) 
print("XGBRegressor 平均方差", val_error)           
plot_predictions([xgb_reg], X, y, axes=[-0.5, 0.5, -0.1, 0.8])
plt.show()


#进行早停止
xgb_reg = xgboost.XGBRegressor()
xgb_reg.fit(X_train, y_train)
y_pred = xgb_reg.predict(X_val)
xgb_reg.fit(X_train, y_train,eval_set=[(X_val, y_val)], early_stopping_rounds=2)

'''
[0]     validation_0-rmse:0.22834
[1]     validation_0-rmse:0.16224
[2]     validation_0-rmse:0.11843
[3]     validation_0-rmse:0.08760
[4]     validation_0-rmse:0.06848
[5]     validation_0-rmse:0.05709
[6]     validation_0-rmse:0.05297
[7]     validation_0-rmse:0.05129
[8]     validation_0-rmse:0.05155  两次不降则提前停止
[9]     validation_0-rmse:0.05211  
'''
y_pred = xgb_reg.predict([X_val[0]])
print("Early Stop XGBRegressor predict : ",y_pred)

plot_predictions([xgb_reg], X, y, axes=[-0.5, 0.5, -0.1, 0.8])
plt.show()