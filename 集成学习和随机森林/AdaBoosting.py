'''提升法'''
'''
AdaBoost
首先训练一个基础分类器，并使用它对训练集进行预测。
然后，该算法会增加分类错误的训练实例的相对权重。
然后，它使用更新后的权重训练第二个分类器，
并再次对训练集进行预测，更新实例权重，
以此类推
Adaboost是一种迭代算法，其核心思想是针对同一个训练集训练不同的分类器(弱分类器)，
然后把这些弱分类器集合起来，
构成一个更强的最终分类器（强分类器）
boosing n.助推  v.推进
'''
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons

from plot_decision_boundary import plot_decision_boundary

#数据准备
X, y = make_moons(n_samples=500, noise=0.30, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
plt.plot(X[:, 0][y==0], X[:, 1][y==0], "yo", alpha=0.5)
plt.plot(X[:, 0][y==1], X[:, 1][y==1], "bs", alpha=0.5)
plt.show()


#训练AdaBoostClassifier模型
'''
SAMME（基于多类指数损失函数的逐步添加模型）
如果预测器可以估算类概率（即具有predict_proba（）方法）
Scikit-Learn会使用一种SAMME的变体，称为SAMME.R（R代表“Real”）
它依赖的是类概率而不是类预测
'''
ada_clf = AdaBoostClassifier(
    DecisionTreeClassifier(max_depth=1), n_estimators=200,
    algorithm="SAMME.R", learning_rate=0.5, random_state=42)
ada_clf.fit(X_train, y_train)
plot_decision_boundary(ada_clf, X, y)
plt.show()



from sklearn.svm import SVC
import numpy as np

m = len(X_train)
fix, axes = plt.subplots(ncols=2, figsize=(10,4), sharey=True)
for subplot, learning_rate in ((0, 1), (1, 0.5)):
    sample_weights = np.ones(m) #len为m的向量[1,1,...]
    plt.sca(axes[subplot])
    #boosting 根据上次训练集在模型上的预测结果进行夹权处理
    for i in range(5):
        #迭代五次SVM分类器
        svm_clf = SVC(kernel="rbf", C=0.05, gamma="scale", random_state=42)
        #自定义样本权重
        svm_clf.fit(X_train, y_train, sample_weight=sample_weights)
        #使用训练集进行预测
        y_pred = svm_clf.predict(X_train)
        #将预测错误的样本的权重增加 1+学习率 倍
        sample_weights[y_pred != y_train] *= (1 + learning_rate)
        plot_decision_boundary(svm_clf, X, y, alpha=0.2)
        plt.title("learning_rate = {}".format(learning_rate), fontsize=16)
plt.show()