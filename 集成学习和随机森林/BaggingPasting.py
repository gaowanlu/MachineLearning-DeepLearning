'''
除了投票器分类(使用不同的模型)
还有另一种方法、使用相同的模型算法，使用不同的训练集进行训练。
采样时如果将
样本放回，这种方法叫作bagging
采样时样本不放回，这种方法则叫作pasting

bagging和pasting都允许训练实例在多个预测器中被多次采样，但是只有
bagging允许训练实例被同一个预测器多次采样

'''

# Scikit-Learn中的bagging和pasting
# BaggingClassifier类  BaggingRegressor用于回归
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np

#数据准备
X, y = make_moons(n_samples=500, noise=0.30, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
plt.plot(X[:, 0][y==0], X[:, 1][y==0], "yo", alpha=0.5)
plt.plot(X[:, 0][y==1], X[:, 1][y==1], "bs", alpha=0.5)
plt.show()

#模型训练
bag_clf = BaggingClassifier(DecisionTreeClassifier(), n_estimators=500,
max_samples=100, bootstrap=True, n_jobs=-1)
#训练了一个包含500个决策树分类器[5]的集成
#每次从训练集中随机采样100个训练实例进行训练
#bootstrap=False => pasting      True=> bagging
bag_clf.fit(X_train, y_train)


y_pred = bag_clf.predict(X_test)
from sklearn.metrics import accuracy_score
print("集成 准确率 ",accuracy_score(y_test, y_pred))

# 训练单个决策树
tree_clf = DecisionTreeClassifier(random_state=42)
tree_clf.fit(X_train, y_train)
y_pred_tree = tree_clf.predict(X_test)
print("单个模型 准确率 ",accuracy_score(y_test, y_pred_tree))


#显示决策树的决策边界
from matplotlib.colors import ListedColormap
from plot_decision_boundary import plot_decision_boundary

plot_decision_boundary(tree_clf, X, y)
plt.show()
plot_decision_boundary(bag_clf, X, y)
plt.show()


#包外评估
'''
使用bagging，有些实例可能会被采样多次，而有些实例则
可能根本不被采样。BaggingClassifier默认采样m个训练实例，然后放回样本
（bootstrap=True），m是训练集的大小。这意味着对每个预测器来说，平均只对63%的训
练实例进行采样。剩余37%未被采样的训练实例称为包外（oob）实例。注意，对所有
预测器来说，这是不一样的37%
每个预测器在训练时只用到了训练集中的一部分,没用到的为包外实例oob
'''
#在创建BaggingClassifier时，将obb_score设为True就可以请求在训练结束时
#进行包外评估
bag_clf=BaggingClassifier(DecisionTreeClassifier(),n_estimators=500,
bootstrap=True,n_jobs=-1,oob_score=True)
bag_clf.fit(X_train,y_train)
print("包外评估 ",bag_clf.oob_score_)
# 包外评估  0.8986666666666666
#BaggingClassifier能在测试集上达到0.89+ 的准确率
#print(bag_clf.oob_decision_function_) 包外决策函数


#随机补丁和随机子空间
'''
BaggingClassifier类支持对特征采样
由 max_features和bootstrap_features控制,工作方式与max_samples和bootstrap相同
用于特征采样而不是实例采样。
每个预测器将用输入特征的随机子集进行训练。

对训练实例和特征都进行抽样，这称为随机补丁方法
保留所有训练实例（即bootstrap=False并且max_samples=1.0）但是对
特征进行抽样（即bootstrap_features=True并且/或max_features<1.0），这被称为随机子空
间法

'''