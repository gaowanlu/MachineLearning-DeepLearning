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

# 常用参数 
print("每个混合元素权重",gm.weights_)
print("每个混合元素均值",gm.means_)
print("每个混合元素的协方差",gm.covariances_)
print("是否收敛",gm.converged_)
print("需要迭代次数",gm.n_iter_) 

# 使用模型预测 
predict_result=gm.predict(X) #使用训练集做测试集

# 画出预测结果
plt.subplot(1,2,2)
plt.scatter(X[:,0],X[:,1],c=predict_result,s=1)
plt.title("predict_result")
plt.show()

print("硬预测",gm.predict(X))  
print("软预测",gm.predict_proba(X))


#这是一个生成式模型，所以你可以从中采样新的实例(并获得它们的标签) 
X_new, y_new = gm.sample(200) #采集200个 
plt.scatter(X_new[:,0],X_new[:,1],c=y_new,s=1)
plt.title("gm.sample")
plt.show()
plot_gaussian_mixture(gm, X)
plt.show()


#相关参数 
# 你可以通过设置' covariance_type '超参数来对算法寻找的协方差矩阵施加约束:
# * ' "full" '(默认):没有约束，所有集群可以采取任何大小的任何椭球形状。
# *“捆绑”:所有簇必须具有相同的形状，可以是任何椭球(即，它们都共享相同的协方差矩阵)。
# *“球形”:所有星团必须是球形的，但它们可以有不同的直径(即不同的方差)。
# * ' "diag" ':星团可以呈现任何大小的椭球形，但椭球的轴线必须平行于轴线
gm_full = GaussianMixture(n_components=3, n_init=10, covariance_type="full", random_state=42)
gm_tied = GaussianMixture(n_components=3, n_init=10, covariance_type="tied", random_state=42)
gm_spherical = GaussianMixture(n_components=3, n_init=10, covariance_type="spherical", random_state=42)
gm_diag = GaussianMixture(n_components=3, n_init=10, covariance_type="diag", random_state=42)
gm_full.fit(X)
plt.subplot(2,2,1)
plt.title('covariance_type="full"')
plot_gaussian_mixture(gm, X)

gm_tied.fit(X)
plt.subplot(2,2,2)
plt.title('covariance_type="tied"')
plot_gaussian_mixture(gm, X)

gm_spherical.fit(X)
plt.subplot(2,2,3)
plt.title('covariance_type="spherical"')
plot_gaussian_mixture(gm, X)

gm_diag.fit(X)
plt.subplot(2,2,4)
plt.title('covariance_type="diag"')
plot_gaussian_mixture(gm, X)

plt.show()