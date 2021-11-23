import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_swiss_roll
from sklearn.decomposition import KernelPCA
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
'''内核PCA主要应对非线性'''

#瑞士卷数据准备
X, t = make_swiss_roll(n_samples=1000, noise=0.2, random_state=42)

rbf_pca = KernelPCA(n_components = 2, kernel="rbf", gamma=0.04)#rbf内核 目标维度 2
#降维
X_reduced = rbf_pca.fit_transform(X)

#不同的内核
lin_pca = KernelPCA(n_components = 2, kernel="linear", fit_inverse_transform=True)
rbf_pca = KernelPCA(n_components = 2, kernel="rbf", gamma=0.0433, fit_inverse_transform=True)
sig_pca = KernelPCA(n_components = 2, kernel="sigmoid", gamma=0.001, coef0=1, fit_inverse_transform=True)


#画图
imgs=((131, lin_pca, "Linear kernel"), (132, rbf_pca, "RBF kernel"), (133, sig_pca, "Sigmoid kernel"))

for subplot, pca, title in imgs:
    X_reduced = pca.fit_transform(X)#降维
    plt.subplot(subplot)
    plt.title(title, fontsize=14)
    plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=t, cmap=plt.cm.hot)#画图
    plt.xlabel("$z_1$", fontsize=18)
    if subplot == 131:
        plt.ylabel("$z_2$", fontsize=18, rotation=0)
    plt.grid(True)
plt.show()


plt.figure(figsize=(6, 5))


X_reduced=rbf_pca.fit_transform(X)

#逆升维
X_inverse = rbf_pca.inverse_transform(X_reduced)


#先降维再训练分类器
clf = Pipeline([
        ("kpca", KernelPCA(n_components=2)),
        ("log_reg", LogisticRegression(solver="lbfgs"))
    ])

param_grid = [{
        "kpca__gamma": np.linspace(0.03, 0.05, 10),#0.03 - 0.05 分10个数
        "kpca__kernel": ["rbf", "sigmoid"]
    }]
#网格搜索
grid_search = GridSearchCV(clf, param_grid, cv=3)



#二分类
y = t > 6.9
grid_search.fit(X, y)
print(grid_search.best_params_)
#{'kpca__gamma': 0.043333333333333335, 'kpca__kernel': 'rbf'}

rbf_pca = KernelPCA(n_components = 2, kernel="rbf", gamma=0.0433,
                    fit_inverse_transform=True)
X_reduced = rbf_pca.fit_transform(X)
X_preimage = rbf_pca.inverse_transform(X_reduced) 

from sklearn.metrics import mean_squared_error#均方误差
print(mean_squared_error(X, X_preimage))#32.78630879576611