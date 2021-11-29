from matplotlib.image import imread
import os
from K_MeansUtil import plot_decision_boundaries
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pylab as plt
from sklearn.cluster import KMeans

# 读取图片
image = imread(os.path.join(".", "ladybug.png"))
print(image.shape)

X = image.reshape(-1, 3)

#训练KMeans
#k=8
#kmeans = KMeans(n_clusters=8, random_state=42).fit(X)
# print(kmeans.cluster_centers_)
# print(kmeans.labels_)
#segmented_img = kmeans.cluster_centers_[kmeans.labels_] #将每个像素点专为 该集群中心点的三通道值
#print(segmented_img)

#segmented_img = segmented_img.reshape(image.shape)


# 不同的k值情况

segmented_imgs = []
#n_colors = (10, 8, 6, 4, 2)
n_colors = (2,)
for n_clusters in n_colors:
    print("train k={}".format(n_clusters))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(X)
    segmented_img = kmeans.cluster_centers_[kmeans.labels_]
    segmented_imgs.append(segmented_img.reshape(image.shape))

# 显示图像
plt.figure(figsize=(10, 5))
plt.subplots_adjust(wspace=0.05, hspace=0.1)

plt.subplot(231)
plt.imshow(image)
plt.title("Original image")
plt.axis('off')

for idx, n_clusters in enumerate(n_colors):
    plt.subplot(232 + idx)
    plt.imshow(segmented_imgs[idx])
    plt.title("{} colors".format(n_clusters))
    plt.axis('off')

plt.show()
