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


plt.title("original")
plt.scatter(X[:,0],X[:,1],c=y,s=1)
plt.show()
# 以上情况使用KMeans的效果并不好 

# 使用高斯混合模型 
gm = GaussianMixture(n_components=4, n_init=10, random_state=42)
gm.fit(X)


# 重要指标  
# 贝叶斯信息标准(BIC)或赤池信息标准(AIC) 
print("Bic",gm.bic(X))
print("Aic",gm.aic(X))


gms_per_k = [GaussianMixture(n_components=k, n_init=10, random_state=42).fit(X)
             for k in range(1, 11)]
bics = [model.bic(X) for model in gms_per_k]
aics = [model.aic(X) for model in gms_per_k]
plt.plot(range(1, 11), bics, "bo-", label="BIC")
plt.plot(range(1, 11), aics, "go--", label="AIC")
plt.xlabel("$k$", fontsize=14)
plt.ylabel("Information Criterion", fontsize=14)
plt.show()


# 搜索集群数量和' covariance_type '超参数值的最佳组合 
min_bic = np.infty
for k in range(1, 11):
    for covariance_type in ("full", "tied", "spherical", "diag"):
        bic = GaussianMixture(n_components=k, n_init=10,
                              covariance_type=covariance_type,
                              random_state=42).fit(X).bic(X)
        if bic < min_bic:
            min_bic = bic
            best_k = k
            best_covariance_type = covariance_type  
print("best_k",best_k)
print("best_covariance_type",best_covariance_type)