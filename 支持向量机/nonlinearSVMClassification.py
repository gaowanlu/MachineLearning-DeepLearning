'''
在许多情况下，现有特征无法做到线性可分离
添加更多特征达到可分离 
例如
.   . x . .
-4 -2 0 2 4  如法做到先行分离
将其平方
16 4 0 4 16
.            .
    .      .
---------------  
        x
x轴为x x轴为x^2 则可变为线性可分
'''
from numpy.lib import polynomial
from sklearn.datasets import make_moons
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures,StandardScaler
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
import numpy as np

'''---------------------数据集准备--------------------------'''
#生成半圆错交数据集
X,y=make_moons(n_samples=100,noise=0.15)
def plot_dataset(X, y, axes):
    plt.plot(X[:, 0][y==0], X[:, 1][y==0], "bs")
    plt.plot(X[:, 0][y==1], X[:, 1][y==1], "g^")
    plt.axis(axes)
    plt.grid(True, which='both')
    plt.xlabel(r"$x_1$", fontsize=20)
    plt.ylabel(r"$x_2$", fontsize=20, rotation=0)
plot_dataset(X, y, [-1.5, 2.5, -1, 1.5])
plt.show()

'''------------------------训练svm模型---------------------------'''
polynomial_svm_clf=Pipeline([
    ("poly_features",PolynomialFeatures(degree=3)),
    ("scaler",StandardScaler()),
    ("svm_clf",LinearSVC(C=10,loss="hinge"))
])
polynomial_svm_clf.fit(X,y)
polynomial_svm_clf.predict([[1.0,-4]])


'''-------------------------预判情况 画图--------------------------'''
#画出分类情况
def plot_predictions(clf, axes):
    x0s = np.linspace(axes[0], axes[1], 100)#横轴范围分割100个
    x1s = np.linspace(axes[2], axes[3], 100)#纵轴范围分割100个
    x0, x1 = np.meshgrid(x0s, x1s) #网格化 像素画
    X = np.c_[x0.ravel(), x1.ravel()]#要测试的每个点
    y_pred = clf.predict(X).reshape(x0.shape)
    y_decision = clf.decision_function(X).reshape(x0.shape)#决策
    plt.contourf(x0, x1, y_pred, cmap=plt.cm.brg, alpha=0.2)
    plt.contourf(x0, x1, y_decision, cmap=plt.cm.brg, alpha=0.1)
#画出分类区域划分
plot_predictions(polynomial_svm_clf, [-1.5, 2.5, -1, 1.5])
plot_dataset(X, y, [-1.5, 2.5, -1, 1.5])#画出所有数据集
plt.show()



#多项式内核
from sklearn.svm import SVC
polynomial_svm_clf=Pipeline([
    ("scaler",StandardScaler()),
    ("svm_clf",SVC(kernel="poly",degree=3,coef0=1,C=5))
])
polynomial_svm_clf.fit(X,y)

#相似特征、高斯RBF


#高斯RBF内核
rbf_kernel_svm_clf=Pipeline([
    ("scaler",StandardScaler()),
    ("svm_clf",SVC(kernel="rbf",gamma=5,C=1000))
])
rbf_kernel_svm_clf.fit(X,y)
plot_predictions(rbf_kernel_svm_clf,[-1.5, 2.5, -1, 1.5])
plot_dataset(X, y, [-1.5, 2.5, -1, 1.5])#画出所有数据集
plt.show()
'''
超参数gamma（γ）和C使用不同值时
的模型。增加gamma值会使钟形曲线变得更窄，因此每个实例的影响
范围随之变小：决策边界变得更不规则，开始围着单个实例绕弯。反过来，减小gamma值
使钟形曲线变得更宽，因而每个实例的影响范围增大，决策边界变得更平坦。所以γ就像
是一个正则化的超参数：模型过拟合，就降低它的值，如果欠拟合则提升它的值（类似超
参数C）
'''

# 计算复杂度
'''
类           时间复杂度        核外支持         需要缩放          核技巧
LinearSVC   O(mxn)             否               是              否
SGDClassifier O(mxn)           是               是              否
SVC          O(m^2 x n)到O(m^3 x n) 否          是              是
'''
