from K_MeansUtil import plot_decision_boundaries
from DBSCANUtil import plot_dbscan
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons
import numpy as np
import matplotlib.pylab as plt



#数据准备
X, y = make_moons(n_samples=1000, noise=0.05, random_state=42)

dbscan = DBSCAN(eps=0.05, min_samples=5)
dbscan.fit(X)
print(dbscan.labels_[:10])
print("核心样本指数 ",len(dbscan.core_sample_indices_))  # 核心样本下标

# core_sample_indices_: 核心样本下标。
# labels_: 数据集中每个点的集合标签给, 噪声点标签为-1。
# components_ ：核心样本的副本

# dbscan.core_sample_indices_[:10]
# dbscan.components_[:3]

print(len(np.unique(dbscan.labels_)))#8

dbscan2 = DBSCAN(eps=0.2)
dbscan2.fit(X)
#DBSCAN 参数
# eps: 两个样本之间的最大距离，即扫描半径
# min_samples ：作为核心点的话邻域(即以其为圆心，eps为半径的圆，含圆上的点)中的最小样本数(包括点本身)。
# 其他参数：
# metric ：度量方式，默认为欧式距离，还有metric =‘precomputed’（稀疏半径邻域图）
# algorithm：近邻算法求解方式，有四种：‘auto’, ‘ball_tree’, ‘kd_tree’, ‘brute’
# leaf_size：叶的大小，在使用BallTree or cKDTree近邻算法时候会需要这个参数
# n_jobs ：使用CPU格式，- 1代表全开


# 画出效果
plt.figure(figsize=(9, 3.2))
plt.subplot(121)
plot_dbscan(dbscan, X, size=100)
plt.subplot(122)
plot_dbscan(dbscan2, X, size=600, show_ylabels=False)
plt.show()


# KNN分类器
dbscan = dbscan2#使用第二个DBCSCAN模型聚类的结果 
knn = KNeighborsClassifier(n_neighbors=50)
#样本与相应相应下标的标签
print("\nKNN训练集标签=> ",dbscan.labels_[dbscan.core_sample_indices_])

knn.fit(dbscan.components_, dbscan.labels_[dbscan.core_sample_indices_])

# 使用KNN模型进行预测
print("使用DBSCAN贴标签 使用KNN训练分类器")
new_test = np.array([[-0.5, 0], [0, 0.5], [1, -0.1], [2, 1]])

print(knn.predict(new_test)) #预测标签
print(knn.predict_proba(new_test))
# predict_proba返回的是一个 n 行 k 列的数组， 
# 第 i 行 第 j 列上的数值是模型预测 第 i 个预测样本为某个标签的概率，并且每一行的概率和为1


plot_decision_boundaries(knn, X, show_centroids=False)
#画出测试集用+
plt.scatter(new_test[:, 0], new_test[:, 1],
            c="b", marker="+", s=200, zorder=10)
plt.show()


y_dist, y_pred_idx = knn.kneighbors(new_test, n_neighbors=1)
print("dist =>",y_dist,"\ncenter_index=> ",y_pred_idx)#距离与类别中心点数据下标
#获得相应标签
y_pred = dbscan.labels_[dbscan.core_sample_indices_][y_pred_idx]

#限制距离
# y_pred[y_dist > 0.2] = -1
# y_pred.ravel() #扁平化
