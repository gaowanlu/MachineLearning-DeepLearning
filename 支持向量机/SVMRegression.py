
'''
不再尝试拟合两个类之间可能的最宽街道
的同时限制间隔违例，SVM回归要做的是让尽可能多的实例位于街道上，同时限制间隔
违例（也就是不在街道上的实例）。街道的宽度由超参数ε控制(epsilon)。
'''
from sklearn.svm import LinearSVR,SVR
from sklearn import datasets
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

#寻找支持向量
def find_support_vectors(svm_reg, X, y):
    y_pred = svm_reg.predict(X)
    off_margin = (np.abs(y - y_pred) >= svm_reg.epsilon)
    return np.argwhere(off_margin)

#画出分界线与间隔
def plot_svm_regression(svm_reg, X, y, axes):
    x1s = np.linspace(axes[0], axes[1], 100).reshape(100, 1)
    y_pred = svm_reg.predict(x1s)
    plt.plot(x1s, y_pred, "k-", linewidth=2, label=r"$\hat{y}$")
    plt.plot(x1s, y_pred + svm_reg.epsilon, "k--")
    plt.plot(x1s, y_pred - svm_reg.epsilon, "k--")
    plt.scatter(X[svm_reg.support_], y[svm_reg.support_], s=180, facecolors='#FFAAAA')
    plt.plot(X, y, "bo")
    plt.xlabel(r"$x_1$", fontsize=18)
    plt.legend(loc="upper left", fontsize=18)
    plt.axis(axes)


def use_LinearSVR():
    #数据准备
    np.random.seed(42)
    m = 50
    X = 2 * np.random.rand(m, 1)
    y = (4 + 3 * X + np.random.randn(m, 1)).ravel()
    #训练模型
    svm_reg=LinearSVR(epsilon=1.5)
    svm_reg.fit(X,y)
    plt.plot(X, y, "bo")
    #画出回归直线
    #寻找支持向量
    svm_reg.support_ = find_support_vectors(svm_reg, X, y)
    plot_svm_regression(svm_reg, X, y, [0, 2, 3, 11])
    plt.show()

#多阶
def use_SVR():
    np.random.seed(42)
    m = 100
    X = 2 * np.random.rand(m, 1) - 1
    y = (0.2 + 0.1 * X + 0.5 * X**2 + np.random.randn(m, 1)/10).ravel()
    svm_poly_reg = SVR(kernel="poly", degree=2, C=100, epsilon=0.1)
    svm_poly_reg.fit(X, y)
    plot_svm_regression(svm_poly_reg, X, y, [-1, 1, 0, 1])
    plt.title(r"$degree={}, C={}, \epsilon = {}$".format(svm_poly_reg.degree, svm_poly_reg.C, svm_poly_reg.epsilon), fontsize=18)
    plt.ylabel(r"$y$", fontsize=18, rotation=0)
    plt.show()

def main():
    use_LinearSVR()
    use_SVR()

if __name__ == '__main__':
    main()

#有许多的知识等待着我们学习 这些只是让我们了解了SVM的基本运作
#加油吧少年！！！