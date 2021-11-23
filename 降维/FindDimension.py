from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


#知道怎么使用PCA进行降维 那降到多少为合适的呢
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train_transed=[]
for index in range(len(x_train)):
    x_train_transed.append(x_train[index].reshape(-1))


pca = PCA()
pca.fit(x_train_transed)
cumsum = np.cumsum(pca.explained_variance_ratio_)#解释方差比
dimension = np.argmax(cumsum >= 0.95) + 1
print(dimension)#154为最低满足0.95得了 再降维的话将小于0.95

#再来一次,指定维度 直接转换数据集进行降维
pca=PCA(n_components=154)
X_reduced=pca.fit_transform(x_train_transed)
print(X_reduced[:5])
print(pca.n_components_)#154
print(np.sum(pca.explained_variance_ratio_))#0.9501960192613029


#可知道PCA 降维时有部分没有被解释到、图像从784维 压缩到了 154维
#利用逆变换 变换回去
x_recovered=pca.inverse_transform(X_reduced)
some_digit_image = x_recovered[0].reshape(28, -1)
plt.imshow(some_digit_image, cmap="binary")
plt.show()


