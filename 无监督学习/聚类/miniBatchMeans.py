from sklearn.cluster import MiniBatchKMeans
from K_MeansUtil import plot_decision_boundaries
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pylab as plt
# from sklearn.cluster import KMeans

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
minibatch_kmeans=MiniBatchKMeans(n_clusters=5)
minibatch_kmeans.fit(X)
print(minibatch_kmeans.inertia_)
'''
1) n_clusters: 即我们的k值，和KMeans类的n_clusters意义一样。

2）max_iter：最大的迭代次数， 和KMeans类的max_iter意义一样。

3）n_init：用不同的初始化质心运行算法的次数。这里和KMeans类意义稍有不同，
   KMeans类里的n_init是用同样的训练集数据来跑不同的初始化质心从而运行算法
   。而MiniBatchKMeans类的n_init则是每次用不一样的采样数据集来跑不同的初始化质心运行算法。

4）batch_size：即用来跑Mini Batch KMeans算法的采样集的大小，默认是100.
   如果发现数据集的类别较多或者噪音点较多，需要增加这个值以达到较好的聚类效果。

5）init： 即初始值选择的方式，和KMeans类的init意义一样。

6）init_size: 用来做质心初始值候选的样本个数，默认是batch_size的3倍，
   一般用默认值就可以了。

7）reassignment_ratio: 某个类别质心被重新赋值的最大次数比例，
   这个和max_iter一样是为了控制算法运行时间的。这个比例是占样本总数的比例，乘以样本总数就得到了每个类别质心可以重新赋值的次数。
   如果取值较高的话算法收敛时间可能会增加，尤其是那些暂时拥有样本数较少的质心。
   默认是0.01。如果数据量不是超大的话，比如1w以下，建议使用默认值。如果数据量超过1w，
   类别又比较多，可能需要适当减少这个比例值。具体要根据训练集来决定。

8）max_no_improvement：即连续多少个Mini Batch没有改善聚类效果的话，
   就停止算法， 和reassignment_ratio， max_iter一样是为了控制算法运行时间的。
   默认是10.一般用默认值就足够了。
   
'''