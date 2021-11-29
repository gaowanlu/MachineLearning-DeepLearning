from K_MeansUtil import plot_decision_boundaries
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pylab as plt
from sklearn.cluster import KMeans


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

# 控制迭代参数
# 初始中心点随机，最大迭代 1 2 3 

kmeans_iter1 = KMeans(n_clusters=5, init="random", n_init=1,
                      algorithm="full", max_iter=1, random_state=1)
kmeans_iter2 = KMeans(n_clusters=5, init="random", n_init=1,
                      algorithm="full", max_iter=2, random_state=1)
kmeans_iter3 = KMeans(n_clusters=5, init="random", n_init=1,
                      algorithm="full", max_iter=3, random_state=1)
kmeans_iter1.fit(X)
kmeans_iter2.fit(X)
kmeans_iter3.fit(X)


plot_decision_boundaries(kmeans_iter1, X)
plt.show()
plot_decision_boundaries(kmeans_iter2, X)
plt.show()
plot_decision_boundaries(kmeans_iter3, X)
plt.show()

# kmeans api 可以初始化中心点  这样可以有效算法的进行   
good_init = np.array([[-3, 3], [-3, 2], [-3, 1], [-1, 2], [0, 2]])
kmeans = KMeans(n_clusters=5, init=good_init, n_init=1)
kmeans.fit(X)
plot_decision_boundaries(kmeans, X)
plt.show()
#模型惯性
print("模型惯性",kmeans.inertia_,"  模型负惯性",kmeans.score)
'''
(1) n_clusters: 即我们的k值，一般需要多试一些值以获得较好的聚类效果。
(2）max_iter： 最大的迭代次数，一般如果是凸数据集的话可以不管这个值，如果数据集不是凸的，
    可能很难收敛，此时可以指定最大的迭代次数让算法可以及时退出循环。
(3）n_init：用不同的初始化质心运行算法的次数。由于K-Means是结果受初始值影响的局部最优的迭代算法，
    因此需要多跑几次以选择一个较好的聚类效果，默认是10，一般不需要改。如果你的k值较大，则可以适当增大这个值。
(4）init： 即初始值选择的方式，可以为完全随机选择'random', 优化过的'k-means++'或者自己指定初始化的k个质心。
    一般建议使用默认的'k-means++'。
(5）algorithm：有“auto”, “full” or “elkan”三种选择。"full"就是我们传统的K-Means算法， 
    “elkan”是我们原理篇讲的elkan K-Means算法。默认的"auto"则会根据数据值是否是稀疏的，
    来决定如何选择"full"和“elkan”。一般数据是稠密的，那么就是 “elkan”，否则就是"full"。
    一般来说建议直接用默认的"auto"
'''
