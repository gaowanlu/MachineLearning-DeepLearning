from sklearn.manifold import MDS
from sklearn.manifold import Isomap
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll
from tensorflow import keras

'''其他降维技术'''

#瑞士卷数据准备
X, t = make_swiss_roll(n_samples=1000, noise=0.2, random_state=41)

#多维缩放
mds = MDS(n_components=2, random_state=42)
X_reduced_mds = mds.fit_transform(X)
#Isomap
isomap = Isomap(n_components=2)
X_reduced_isomap = isomap.fit_transform(X)
#t分布随机近邻嵌入
tsne = TSNE(n_components=2, random_state=42)
X_reduced_tsne = tsne.fit_transform(X)
#线性判别分析
lda = LinearDiscriminantAnalysis(n_components=2) 

#mnist 准备 tf => sklearn
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train_transed=[]
for index in range(len(x_train)):
    x_train_transed.append(x_train[index].reshape(-1))

lda.fit(x_train_transed, y_train)
X_reduced_lda = lda.transform(x_train_transed)


#画图
titles = ["MDS", "Isomap", "t-SNE"]
plt.figure(figsize=(11,4))

for subplot, title, X_reduced in zip((131, 132, 133), titles,
                                     (X_reduced_mds, X_reduced_isomap, X_reduced_tsne)):
    plt.subplot(subplot)
    plt.title(title, fontsize=14)
    plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=t, cmap=plt.cm.hot)
    plt.xlabel("$z_1$", fontsize=18)
    if subplot == 131:
        plt.ylabel("$z_2$", fontsize=18, rotation=0)
    plt.grid(True)

plt.show()