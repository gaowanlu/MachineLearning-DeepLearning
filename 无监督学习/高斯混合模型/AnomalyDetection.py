# 使用高斯混合模型 进行异常值检测 
from sklearn.mixture import GaussianMixture
import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pylab as plt
from gaussianMixtureUtil import plot_gaussian_mixture;
# 数据准备 
X1, y1 = make_blobs(n_samples=1000, centers=((4, -4), (0, 0)), random_state=42)
X1 = X1.dot(np.array([[0.374, 0.95], [0.732, 0.598]]))
print("X1",X1)
X2, y2 = make_blobs(n_samples=250, centers=1, random_state=42)
X2 = X2 + [6, -8] #偏差
X = np.r_[X1, X2]
y = np.r_[y1, y2]


plt.subplot(1,2,1)
plt.title("original")
plt.scatter(X[:,0],X[:,1],c=y,s=1)

# 以上情况使用KMeans的效果并不好 

# 使用高斯混合模型 
gm = GaussianMixture(n_components=4, n_init=10, random_state=42)
gm.fit(X)



# 异常值检测 
'''
高斯混合可用于异常检测:位于低密度区域的实例可视为异常。
您必须定义您想要使用的密度阈值。
例如，在一个试图检测缺陷产品的制造公司中，
缺陷产品的比例通常是众所周知的。假设它等于4%，
那么你可以将密度阈值设置为导致有4%的实例位于该阈值密度以下的区域的值
'''

densities = gm.score_samples(X)#每个点的区域密度
density_threshold = np.percentile(densities, 4)
anomalies = X[densities < density_threshold]

plt.subplot(1,2,2)
plot_gaussian_mixture(gm, X)
plt.scatter(anomalies[:, 0], anomalies[:, 1], color='r', marker='*')
plt.ylim(top=5.1)
plt.show()