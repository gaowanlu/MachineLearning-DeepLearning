import numpy as np
import matplotlib.pyplot as plt

#----------数据准备
np.random.seed(42)
m = 200
X = np.random.rand(m, 1)
y = 4 * (X - 0.5) ** 2
y = y + np.random.randn(m, 1) / 10
plt.plot(X,y,"r.");
plt.show()


#----------训练决策树回归模型
from sklearn.tree import DecisionTreeRegressor
#正则
tree_reg1 = DecisionTreeRegressor(random_state=42, max_depth=2)
tree_reg2 = DecisionTreeRegressor(random_state=42, max_depth=3)
tree_reg1.fit(X, y)
tree_reg2.fit(X, y)


#----------展示模型效果
def plot_regression_predictions(tree_reg, X, y, axes=[0, 1, -0.2, 1], ylabel="$y$"):
    x1 = np.linspace(axes[0], axes[1], 500).reshape(-1, 1)
    y_pred = tree_reg.predict(x1)
    plt.axis(axes)
    plt.xlabel("$x_1$", fontsize=18)
    if ylabel:
        plt.ylabel(ylabel, fontsize=18, rotation=0)
    plt.plot(X, y, "b.")
    plt.plot(x1, y_pred, "r.-", linewidth=2, label=r"$\hat{y}$")



plot_regression_predictions(tree_reg1, X, y)
plt.show()
plot_regression_predictions(tree_reg2, X, y, ylabel=None)
plt.show()


#无正则
tree_reg3 = DecisionTreeRegressor( random_state=42)
tree_reg3.fit(X, y)
plot_regression_predictions(tree_reg3, X, y, ylabel=None)
plt.show()


#限制叶子节点的最小样本数量
tree_reg4 = DecisionTreeRegressor(random_state=42, min_samples_leaf=7)
tree_reg4.fit(X, y)
plot_regression_predictions(tree_reg4, X, y, ylabel=None)
plt.show()

#保存决策树用dot可视化
from graphviz import Source
from sklearn.tree import export_graphviz
import os
export_graphviz(
        tree_reg1,
        out_file=os.path.join(".", "regression_tree.dot"),
        feature_names=["x1"],
        rounded=True,
        filled=True
    )
Source.from_file(os.path.join(".", "regression_tree.dot"))


