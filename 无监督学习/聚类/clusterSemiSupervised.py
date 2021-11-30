# 使用聚类进行半监督学习
# 集群的另一个用例是半监督学习，当我们有大量的未标记实例和很少的标记实例时
# 数据准备
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


X_digits, y_digits = load_digits(return_X_y=True)
print(len(X_digits))  # 1797张手写数字数据集
print(y_digits)
X_train, X_test, y_train, y_test = train_test_split(
    X_digits, y_digits, random_state=42)

# 先取50个训练集
n_labeled = 50

log_reg = LogisticRegression(
    multi_class="ovr", solver="lbfgs", random_state=42)
log_reg.fit(X_train[:n_labeled], y_train[:n_labeled])
print("贴标签 n=50 精准率 ", log_reg.score(X_test, y_test))
# 精准率为0.83

# 使用聚类
k = 50 # 为什么用50 因为每个图像50个像素
kmeans = KMeans(n_clusters=k, random_state=42)
X_digits_dist = kmeans.fit_transform(X_train)  # 每个簇都转化为中心点的值

representative_digit_idx = np.argmin(X_digits_dist, axis=0)
# 得到每个列中的最小值 的列下标
print(representative_digit_idx)
#得到训练集中每个像素点值最小的一个数据集
X_representative_digits = X_train[representative_digit_idx]
print(len(X_representative_digits))#50 


#显示这些图像  
plt.figure(figsize=(8, 2))
for index, X_representative_digit in enumerate(X_representative_digits):
    plt.subplot(k // 10, 10, index + 1)
    plt.imshow(X_representative_digit.reshape(8, 8), cmap="binary", interpolation="bilinear")
    plt.axis('off') #是否开显示坐标

plt.show()  

#为这些图像贴标签
y_representative_digits = np.array([
    4, 8, 0, 6, 8, 3, 7, 7, 9, 2,
    5, 5, 8, 5, 2, 1, 2, 9, 6, 1,
    1, 6, 9, 0, 8, 3, 0, 7, 4, 1,
    6, 5, 2, 4, 1, 8, 6, 3, 9, 2,
    4, 2, 9, 4, 7, 6, 2, 3, 1, 1])  

#使用这些特殊的图像进行国逻辑回归
log_reg = LogisticRegression(multi_class="ovr", solver="lbfgs", max_iter=5000, random_state=42)
log_reg.fit(X_representative_digits, y_representative_digits)
print("聚类 =》 使用每个像素最小的数据集 =》 精准率 ",log_reg.score(X_test, y_test))
#精准率达到了0.92  


# 如果将这些标签 标签传播到自己所在的集群内 会怎样呢
y_train_propagated = np.empty(len(X_train), dtype=np.int32) 
#标签传播
for i in range(k):
    y_train_propagated[kmeans.labels_==i] = y_representative_digits[i]

#在进行逻辑回归
log_reg = LogisticRegression(multi_class="ovr", solver="lbfgs", max_iter=5000, random_state=42)
log_reg.fit(X_train, y_train_propagated)
print("标签传播=>精准率 ",log_reg.score(X_test, y_test))


#标签传播到集群所有数据可能会一些离群值签了标签
#控制 只把标签传播到最靠近质心的第20个百分位数
percentile_closest = 20

#计算每个数据集到其所在集群质心距离
X_cluster_dist = X_digits_dist[np.arange(len(X_train)), kmeans.labels_]

for i in range(k):
    in_cluster = (kmeans.labels_ == i)
    cluster_dist = X_cluster_dist[in_cluster]
    cutoff_distance = np.percentile(cluster_dist, percentile_closest)
    above_cutoff = (X_cluster_dist > cutoff_distance)
    X_cluster_dist[in_cluster & above_cutoff] = -1

#排除没有贴上标签的部分
partially_propagated = (X_cluster_dist != -1)
X_train_partially_propagated = X_train[partially_propagated]
y_train_partially_propagated = y_train_propagated[partially_propagated]


log_reg = LogisticRegression(multi_class="ovr", solver="lbfgs", max_iter=5000, random_state=42)
log_reg.fit(X_train_partially_propagated, y_train_partially_propagated)

print("控制标签传播距离 精准率=> ",log_reg.score(X_test, y_test))

print(np.mean(y_train_partially_propagated == y_train[partially_propagated]))
#print(y_train_partially_propagated == y_train[partially_propagated])
#False 0 True 1
# mean() 求取均值  
# 贴标签正确率为0.98+