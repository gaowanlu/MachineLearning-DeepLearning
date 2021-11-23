from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


#知道怎么使用PCA进行降维 那降到多少为合适的呢
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train_transed=[]
for index in range(len(x_train)):
    x_train_transed.append(x_train[index].reshape(-1))

#随机PCA 比SVD快 
pca=PCA(n_components=154,svd_solver="randomized")
X_reduced=pca.fit_transform(x_train_transed)
print(pca.components_)
'''
默认情况下，svd_solver实际上设置为"auto"：如果m或n大于500并且d小于m或n的
80%，则Scikit-Learn自动使用随机PCA算法，否则它将使用完全的SVD方法。如果要强制
Scikit-Learn使用完全的SVD，可以将svd_solver超参数设置为"full"。
'''