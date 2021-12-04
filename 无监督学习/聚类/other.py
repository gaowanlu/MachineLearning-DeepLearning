'''其他聚类算法
聚集聚类、BIRCH、均值漂移、相似性传播、谱聚类
'''
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering#谱聚类
from sklearn.cluster import AgglomerativeClustering  # 层次聚类
from sklearn.datasets import make_moons
#数据准备
X, y = make_moons(n_samples=1000, noise=0.05, random_state=42)

# 训练两个不同gamma值的谱聚类模型 
sc1 = SpectralClustering(n_clusters=2, gamma=100, random_state=42)
sc1.fit(X)
sc2 = SpectralClustering(n_clusters=2, gamma=1, random_state=42)
sc2.fit(X)

np.percentile(sc1.affinity_matrix_, 95) #从小到大 95%的数值

# 画出集群标签 

def plot_spectral_clustering(sc, X):
    # 使用聚类后产生的标签sc.labels_
    #根据labels将散点划为不同颜色进行显示 
    plt.scatter(X[:, 0], X[:, 1], marker='.',s=10, c=sc.labels_, cmap="Paired")


#1 行 2 列 第1个位置
plt.subplot(1, 2, 1)
plot_spectral_clustering(sc1, X)
plt.title("RBF gamma={}".format(sc1.gamma), fontsize=14)
#1 行 2 列 第2个位置
plt.subplot(1, 2, 2)
plot_spectral_clustering(sc2, X)
plt.title("RBF gamma={}".format(sc2.gamma), fontsize=14)
plt.show()



# 使用层次聚类
agg = AgglomerativeClustering(linkage="complete").fit(X)
print(agg.children_)
print("AgglomerativeClustering  labels : ",agg.labels_)
print(agg.n_leaves_)
plot_spectral_clustering(agg, X)
plt.title("AgglomerativeClustering", fontsize=14)
plt.show()
