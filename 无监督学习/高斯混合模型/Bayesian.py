from sklearn.mixture import BayesianGaussianMixture 
import matplotlib.pylab as plt
from gaussianMixtureUtil import plot_gaussian_mixture;
import numpy as np
from sklearn.datasets import make_moons
from sklearn.datasets import make_blobs

# 数据准备 
X1, y1 = make_blobs(n_samples=1000, centers=((4, -4), (0, 0)), random_state=42)
X1 = X1.dot(np.array([[0.374, 0.95], [0.732, 0.598]]))
X2, y2 = make_blobs(n_samples=250, centers=1, random_state=42)
X2 = X2 + [6, -8] #偏差
X = np.r_[X1, X2]
y = np.r_[y1, y2]


plt.title("original data")
plt.scatter(X[:,0],X[:,1],c=y,s=1)
plt.show()



bgm = BayesianGaussianMixture(n_components=10, n_init=10, random_state=42)
bgm.fit(X)
print("np.round(bgm.weights_, 2)",np.round(bgm.weights_, 2)) #两位小数 四舍五入
plt.title("BayesianGaussianMixture")
plot_gaussian_mixture(bgm, X) #画出决策边界 
plt.show()



# 权重浓度先验优化 

bgm_low = BayesianGaussianMixture(n_components=10, max_iter=1000, n_init=1,
                                  weight_concentration_prior=0.01, random_state=42)#权重浓度先验参数
bgm_high = BayesianGaussianMixture(n_components=10, max_iter=1000, n_init=1,
                                  weight_concentration_prior=10000, random_state=42)

bgm_low.fit(X)
bgm_high.fit(X)


plt.subplot(1,2,1)
plot_gaussian_mixture(bgm_low, X)
plt.title("weight_concentration_prior = 0.01", fontsize=14)

plt.subplot(1,2,2)
plot_gaussian_mixture(bgm_high, X, show_ylabels=False)
plt.title("weight_concentration_prior = 10000", fontsize=14)


plt.show()

# 月亮数据集 

X_moons, y_moons = make_moons(n_samples=1000, noise=0.05, random_state=42) 
bgm = BayesianGaussianMixture(n_components=10, n_init=10, random_state=42) 
bgm.fit(X_moons)

plt.title("moons")
plt.xlabel("$x_1$", fontsize=14)
plt.ylabel("$x_2$", fontsize=14, rotation=0)
plot_gaussian_mixture(bgm, X_moons, show_ylabels=False)
plt.show()