'''
主成分分析 首先它先找到最接近数据集分布的超平面
然后将所有的数据都投影到找个超平面
'''
import numpy as np

#数据准备
np.random.seed(4)
m = 60
w1, w2 = 0.1, 0.3
noise = 0.1
angles = np.random.rand(m) * 3 * np.pi / 2 - 0.5
X = np.empty((m, 3))
X[:, 0] = np.cos(angles) + np.sin(angles)/2 + noise * np.random.randn(m) / 2
X[:, 1] = np.sin(angles) * 0.7 + noise * np.random.randn(m) / 2
X[:, 2] = X[:, 0] * w1 + X[:, 1] * w2 + noise * np.random.randn(m)

#numpy PCA
X_centered=X-X.mean(axis=0)
print("原始数据集 : \n",X_centered[:5])
U,s,Vt=np.linalg.svd(X_centered)
c1=Vt.T[:,0]
c2=Vt.T[:,1]
print("主成分 : ")
print("c1",c1)
print("c2",c2)
'''
c1 [0.93636116 0.29854881 0.18465208]
c2 [-0.34027485  0.90119108  0.2684542 ]
'''
m, n = X.shape

S = np.zeros(X_centered.shape)
S[:n, :n] = np.diag(s)
np.allclose(X_centered, U.dot(S).dot(Vt))
W2 = Vt.T[:, :2]
X2D = X_centered.dot(W2)# 矩阵变换
X2D_using_svd = X2D
print("NumPy PCA : \n",X2D_using_svd[:5])# 转为二维的结果 
#如何使用SVD方法计算解释的方差比（s是矩阵s的对角线)
print("解释方差比 : ",np.square(s) / np.square(s).sum())
#[0.84248607 0.14631839 0.01119554] 0.01119554第三个解释程度


#sklearn PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
X2D = pca.fit_transform(X)
print("sklearn PCA : \n",X2D[:5])


print(np.allclose(X2D, -X2D_using_svd))


#利用PCA 将多维转为了2D 怎讲将2D再转回多维呢
X3D_inv = pca.inverse_transform(X2D)
print(np.allclose(X3D_inv, X))#False 二者是有差异的
#在投影步骤中会丢失一些信息，因此恢复的3D点与原始3D点并不完全相等 但近似是相等的
#sklearn PCA 可以访问主成分
print("主成分 : ",pca.components_)
#解释方差比
print("解释方差比 ： ",pca.explained_variance_ratio_)
# [0.84248607 0.14631839]
#第一个维度解释了84.2%的方差，而第二个维度解释了14.6%。 第三个结实程度可以利用1-sum来进行计算
print(1-pca.explained_variance_ratio_.sum())
#0.011195535570688975  
