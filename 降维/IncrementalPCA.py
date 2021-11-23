'''
增量PCA
'''
from sklearn.decomposition import IncrementalPCA
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


#知道怎么使用PCA进行降维 那降到多少为合适的呢
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train_transed=[]
for index in range(len(x_train)):
    x_train_transed.append(x_train[index].reshape(-1))


#使用增量PCA的优点是 数据集不用一次性进入内存  
n_batches = 100
inc_pca = IncrementalPCA(n_components=154)
for X_batch in np.array_split(x_train_transed, n_batches):
    print(".", end="") 
    inc_pca.partial_fit(X_batch)#将其馈送到 IncrementalPCA

X_reduced = inc_pca.transform(x_train_transed)

print(X_reduced[0])


'''使用numpy的 memmap()  将数据集送到外存 但可以像内存一样进行使用'''
filename = "my_mnist.data"
m, n = len(x_train_transed),len(x_train_transed[0])
X_mm = np.memmap(filename, dtype='float32', mode='write', shape=(m, n))
X_mm[:] = x_train_transed
del X_mm
X_mm = np.memmap(filename, dtype="float32", mode="readonly", shape=(m, n))
batch_size = m // n_batches
inc_pca = IncrementalPCA(n_components=154, batch_size=batch_size)
inc_pca.fit(X_mm)
rnd_pca = PCA(n_components=154, svd_solver="randomized", random_state=42)#随机PCA
X_reduced = rnd_pca.fit_transform(x_train_transed)
