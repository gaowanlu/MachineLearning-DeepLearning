# 如何寻找最佳的K值 也就是集群的数量  
# 使用轮廓分数 

from sklearn.cluster import MiniBatchKMeans
from K_MeansUtil import plot_decision_boundaries
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pylab as plt
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import silhouette_score

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


# 小批量kmeans
silhouette_scores= []
calinski_harabasz_scores=[]
for i in range(2,10):
    minibatch_kmeans = MiniBatchKMeans(n_clusters=i)
    minibatch_kmeans.fit(X)
    #print(minibatch_kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X, minibatch_kmeans.labels_))
    calinski_harabasz_scores.append(calinski_harabasz_score(X, minibatch_kmeans.labels_))
plt.plot(range(2, 10), silhouette_scores, "bo-")
plt.show()
plt.plot(range(2, 10), calinski_harabasz_scores, "ro-")
plt.show()
