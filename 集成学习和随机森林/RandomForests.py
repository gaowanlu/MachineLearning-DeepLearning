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


# 训练随机森林
'''
500棵树的随机森林分类器
（每棵树限制为最多16个叶节点）
'''
from sklearn.ensemble import RandomForestClassifier
rnd_clf = RandomForestClassifier(n_estimators=500, max_leaf_nodes=16, n_jobs=-1)
rnd_clf.fit(X_train, y_train)
y_pred_rf = rnd_clf.predict(X_test)


print("准确率 ",np.sum(y_test == y_pred_rf) / len(y_test))
#0.92

#RandomForestClassifier具有DecisionTreeClassifier的所有超参数（以控制
#树的生长方式），以及BaggingClassifier的所有超参数来控制集成本身 

'''
随机森林在树的生长上引入了更多的随机性：分裂节点时不再是搜索最好的特征，
而是在一个随机生成的特征子集里搜索最好的特征。这导致决策树具有更大的
多样性，（再一次）用更高的偏差换取更低的方差，总之产生了一个整体性能更优
的模型。
'''
bag_clf = BaggingClassifier(
DecisionTreeClassifier(splitter="random", max_leaf_nodes=16),
n_estimators=500, max_samples=1.0, bootstrap=True, n_jobs=-1)


# 极端随机树
'''
在随机森林里单棵树的生长过程中，每个节点在分裂时仅考虑到了一个随
机子集所包含的特征。如果我们对每个特征使用随机阈值，而不是搜索得出的最佳阈值
（如常规决策树），则可能让决策树生长得更加随机
ExtraTreesClassifier类与RandomForestClassifier类 API 相同
ExtraTreesRegressor类与RandomForestRegressor类的API相同
'''
from sklearn.ensemble import ExtraTreesClassifier
ext_clf = ExtraTreesClassifier(n_estimators=500, max_leaf_nodes=16, n_jobs=-1)
ext_clf.fit(X_train, y_train)
y_pred_rf = ext_clf.predict(X_test)
print("机端随机树准确率 ",np.sum(y_test == y_pred_rf) / len(y_test))


#特征重要性
#sklean 查看使用该特征的树节点平均，减少不纯度的程度衡量特征的重要性
#是一个加权平均值，其中每个节点的权重等于与其关联的训练样本的数量
#skelan自动判断特征的权重计算分数，进行特征缩放，是所有重要性为1
from sklearn.datasets import fetch_openml
import matplotlib
from tensorflow import keras


#mnist数据准备
def get_mnist_data():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    some_digit = x_train[0]
    some_digit_image = some_digit.reshape(28, 28)
    y_train = y_train.astype(np.uint8)
    y_test = y_test.astype(np.uint8)
    #转变数据集形式
    x_train_transed=[]
    for index in range(len(x_train)):
        x_train_transed.append(x_train[index].reshape(-1))
    #将数字标签转换为bool型标签,List内item的转换
    #y_train_5 = (y_train == 5)
    return (x_train_transed,y_train)

(X_train,y_train)=get_mnist_data()

#训练模型
rnd_clf = RandomForestClassifier(n_estimators=100, random_state=42)
rnd_clf.fit(X_train, y_train)
#显示结果
def plot_digit(data):
    image = data.reshape(28, 28)
    plt.imshow(image, cmap = matplotlib.cm.hot,
               interpolation="nearest")
    plt.axis("off")
plot_digit(rnd_clf.feature_importances_)# rnd_clf.feature_importances_为每个特征的重要性
# cbar = plt.colorbar(ticks=[rnd_clf.feature_importances_.min(), rnd_clf.feature_importances_.max()])
# cbar.ax.set_yticklabels(['Not important', 'Very important'])
plt.show()