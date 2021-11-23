'''
LLE
局部线性嵌入（LLE）是另一种强大的非线性降维（NLDR）技术。它是一种流形学习技术。
LLE的工作原理是首先测量每个
训练实例如何与其最近的邻居（c.n.）线性相关，然后寻找可以最好地保留这些局部关系
的训练集的低维表示形式
'''
import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll
from sklearn.manifold import LocallyLinearEmbedding
#瑞士卷数据准备
X, t = make_swiss_roll(n_samples=1000, noise=0.2, random_state=41)
lle = LocallyLinearEmbedding(n_components=2, n_neighbors=10, random_state=42)
X_reduced = lle.fit_transform(X)#降维 瑞士卷展开

#画图
plt.title("LLE", fontsize=14)
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=t, cmap=plt.cm.hot)
plt.grid(True)
plt.show()
