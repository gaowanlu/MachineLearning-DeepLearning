from K_MeansUtil import plot_decision_boundaries
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pylab as plt
from sklearn.cluster import KMeans 
import tensorflow as tf
print("GPU",tf.test.is_gpu_available())
# 数据准备
blob_centers = np.array(
    [[0.2,  2.3],
     [-1.5,  2.3],
     [-2.8,  1.8],
     [-2.8,  2.8],
     [-2.8,  1.3]])
blob_std = np.array([0.4, 0.3, 0.1, 0.1, 0.1])
X, y = make_blobs(n_samples=2000, centers=blob_centers,
                  cluster_std=blob_std, random_state=7)
print(X, y)
plt.scatter(X[:, 0], X[:, 1], c=y, s=1)
plt.text(-3, 40, X, size=15)
plt.show()

# 训练K-Means模型与预测
k = 5
kmeans = KMeans(n_clusters=k, random_state=42)
y_pred = kmeans.fit_predict(X)
print(y_pred)#可见被分成了五类
print(kmeans.labels_)#训练集被贴的标签
print(kmeans.cluster_centers_)#每个类别的中心向量

#标出分类的序号
plt.scatter(X[:, 0], X[:, 1], c=y, s=1)
for x, y in kmeans.cluster_centers_:
    plt.text(x, y,kmeans.predict([[x,y]])[0], size=15)
plt.show()

#使用模型进行预测
X_test = np.array([[-1, 10], [2, 4], [-3, 3.7]])
print("使用模型预测",kmeans.predict(X_test))

#决策边界
# plt.figure(figsize=(8, 4))
plot_decision_boundaries(kmeans, X)
plt.show()

#硬聚类与软聚类
#硬聚类将每个数据都分类某个集群
#软聚类提供所有集群的相似度或者称为距离
print("软聚类",kmeans.transform(X_test))
'''
软聚类 [[8.39492054 7.83825117 7.42335512 7.72825692 8.88352348]
 [5.28320241 2.50035266 4.94171501 3.8674245  5.50719129]
 [1.90891962 3.51890636 0.92701828 2.08579125 2.40746487]]
'''
