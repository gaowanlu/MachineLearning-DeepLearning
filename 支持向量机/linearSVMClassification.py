# 支持向量机三宝
'''
间隔、对偶、核技巧
'''
# 大间隔分类
from sklearn.svm import SVC
from sklearn import datasets
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


'''-----------------数据准备------------------'''
iris=datasets.load_iris()#加载数据集
X=iris["data"][:,(2,3)] 
#花瓣长与宽两个特征   选取全部行的2列与3列的属性
y=iris["target"] #标签
#标签转化
setosa_or_versicolor = (y==0)|(y==1) #y标签中，如果==0 or ==1 则将其元素为True 否则为 False

#时True则选中，False不选中
#选出维吉尼亚鸢尾 与 变色鸢尾 
X=X[setosa_or_versicolor]
y=y[setosa_or_versicolor]

#划分测试集与训练集
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2)


'''----------------训练线性SVM模型---------'''
#SVM Classifier model
svm_clf=SVC(kernel="linear",C=float("inf"))
svm_clf.fit(x_train,y_train)


'''----------------使用模型----------------'''
print(svm_clf.predict([[5.5,1.7]]))
#画出分界线与间隔
def plot_svc_decision_boundary(svm_clf, xmin, xmax):
    w = svm_clf.coef_[0]
    b = svm_clf.intercept_[0]

    # At the decision boundary, w0*x0 + w1*x1 + b = 0
    # => x1 = -w0/w1 * x0 - b/w1
    x0 = np.linspace(xmin, xmax, 200)
    decision_boundary = -w[0]/w[1] * x0 - b/w[1]

    margin = 1/w[1]
    gutter_up = decision_boundary + margin
    gutter_down = decision_boundary - margin

    svs = svm_clf.support_vectors_
    plt.scatter(svs[:, 0], svs[:, 1], s=180, facecolors='#FFAAAA')
    plt.plot(x0, decision_boundary, "k-", linewidth=2)
    plt.plot(x0, gutter_up, "k--", linewidth=2)
    plt.plot(x0, gutter_down, "k--", linewidth=2)
#绘图
plt.plot(x_train[:, 0][y_train==1], x_train[:, 1][y_train==1], "yo",label="versicoclor")
plt.plot(x_train[:, 0][y_train==0], x_train[:, 1][y_train==0], "bs",label="setosa")
plt.legend(loc="upper left", fontsize=14)
plt.xlabel("length", fontsize=14)
plt.ylabel("width", fontsize=14)
plot_svc_decision_boundary(svm_clf, 1, 6.99)
plt.show()


'''------------------------------特征缩放-------------------------'''
#未进行特征缩放
Xs = np.array([[1, 50], [5, 20], [3, 80], [5, 60]]).astype(np.float64)
ys = np.array([0, 0, 1, 1])
svm_clf = SVC(kernel="linear", C=100)
svm_clf.fit(Xs, ys)
plot_svc_decision_boundary(svm_clf, 0, 6)
plt.plot(Xs[:, 0][ys==1], Xs[:, 1][ys==1], "bo")
plt.plot(Xs[:, 0][ys==0], Xs[:, 1][ys==0], "ms")
plt.show()

#进行特征缩放
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X_scaled=scaler.fit_transform(Xs)
svm_clf.fit(X_scaled,ys)
plot_svc_decision_boundary(svm_clf, -2, 2)
plt.plot(X_scaled[:, 0][ys==1], X_scaled[:, 1][ys==1], "bo")
plt.plot(X_scaled[:, 0][ys==0], X_scaled[:, 1][ys==0], "ms")
plt.show()

#软间隔分类 如果SVM模型过拟合，可以尝试通过降低C来对其进行正则化


'''---------------------------模型评估-----------------------------'''
#获得混淆矩阵
#混淆矩阵
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
y_test_pred=cross_val_predict(svm_clf,x_test,y_test,cv=3)
#cross_val_predict同样执行k-折交叉验证，返回的不是评估分数，而是每个折叠的预测
#这意味着对于每个实例都可以得到一个干净的预测
# (干净的意思是模型预测时使用的数据在其训练期间从未见过)
matrix=confusion_matrix(y_test,y_test_pred)
print("混淆矩阵:\n",matrix)
'''
可见在测试集上的效果非常不错
[[13  0]
 [ 0  7]]'''
from sklearn.metrics import precision_score,recall_score,f1_score
precisionScore=precision_score(y_test,y_test_pred)
recallScore=recall_score(y_test,y_test_pred)
f1Score=f1_score(y_test,y_test_pred)
print("测试集上的 : 精度：",precisionScore,"召回率：",recallScore,"F1 ：",f1Score)

#也可以试试训练集上面的效果，共同评测模型的效果