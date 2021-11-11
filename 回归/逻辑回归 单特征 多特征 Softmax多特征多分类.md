# 逻辑回归
> - 估计概率公式
> ![在这里插入图片描述](https://img-blog.csdnimg.cn/3c999702e4d641b6a27b6f1b954b7584.png)
> - 逻辑函数(数值->逻辑值)  
> ![在这里插入图片描述](https://img-blog.csdnimg.cn/ed813fbecbb849859da6f3a2cd524d0b.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBAd2FubHVOMQ==,size_20,color_FFFFFF,t_70,g_se,x_16)
> - 逻辑回归模型预测  
> ![在这里插入图片描述](https://img-blog.csdnimg.cn/7eb7916fd52b4eb4859117b129548248.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBAd2FubHVOMQ==,size_20,color_FFFFFF,t_70,g_se,x_16)
> ![在这里插入图片描述](https://img-blog.csdnimg.cn/9fafb8cf048e46de97c13576c2873a51.png)
> > 当概率越靠近1，则-log(t) 越靠近0，当p越靠近0，-log(t)则越大  
> ![在这里插入图片描述](https://img-blog.csdnimg.cn/740e13f78fa54d728b4284e192368e7e.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBAd2FubHVOMQ==,size_20,color_FFFFFF,t_70,g_se,x_16)

> - 逻辑回归成本函数(对数损失)
> ![在这里插入图片描述](https://img-blog.csdnimg.cn/b522c9edc603449c9784e40cb4b0d6c1.png)
> 偏导
> ![在这里插入图片描述](https://img-blog.csdnimg.cn/2c7f453cc7d5443782fbdda5ae2a24bc.png)
> 

```python
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
```


## 单特征
```python
iris = datasets.load_iris()
print(iris["data"][0])
X=iris["data"][:,3:] #petal width 花瓣宽度,选择每行的第四个(每行共四个)
#[:,3:] 前面选择行范围  后面选择列范围
print(X[0])
y=(iris["target"]==2).astype(np.int)#标签，数组元素为2则元素为1，否则为0

#训练一个逻辑回归模型
from sklearn.linear_model import LogisticRegression
log_reg=LogisticRegression()#创建实例
log_reg.fit(X,y)


X_new = np.linspace(0, 3, 1000).reshape(-1, 1)#生成0 - 3 的等差数列，数列有1000个元素

y_proba = log_reg.predict_proba(X_new)#得出预测概率数组
plt.plot(X_new, y_proba[:, 1], "g-", label="Iris virginica")#是的概率
plt.plot(X_new, y_proba[:, 0], "b--", label="Not Iris virginica")#不是的概率

#使用模型进行预测
print( "predict:",log_reg.predict([[1.7], [1.5]]))
plt.show()
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/f2ed8c0aff76455ea3ab72277bd2ada4.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBAd2FubHVOMQ==,size_20,color_FFFFFF,t_70,g_se,x_16)


## 多特征（二分类）
```python
#使用花两个数据，花的长度与宽度
X = iris["data"][:, (2, 3)] 
y = (iris["target"] == 2).astype(np.int)
log_reg = LogisticRegression(solver="lbfgs", C=10**10, random_state=42)
#控制Scikit-Learn LogisticRegression模型的正则化强度的超参数不是alpha（与其他
#线性模型一样），而是反值C。C值越高，对模型的正则化越少。
log_reg.fit(X, y)
#生成随机测试集
x0, x1 = np.meshgrid(
        np.linspace(2.9, 7, 500).reshape(-1, 1),#长度
        np.linspace(0.8, 2.7, 200).reshape(-1, 1),#宽度
)
#.ravel多维数组转一维
X_new = np.c_[x0.ravel(), x1.ravel()]
y_proba = log_reg.predict(X)
print(y_proba)
```


## Softmax回归(多特征多分类)
```python
X = iris["data"][:, (2, 3)]  # 两种特征长度宽度
y = iris["target"] 

softmax_reg = LogisticRegression(multi_class="multinomial",solver="lbfgs", C=10, random_state=42)
softmax_reg.fit(X, y)

x0, x1 = np.meshgrid(
        np.linspace(0, 8, 500).reshape(-1, 1),
        np.linspace(0, 3.5, 200).reshape(-1, 1),
    )
X_new = np.c_[x0.ravel(), x1.ravel()]


y_proba = softmax_reg.predict_proba(X_new)
print(y_proba)
y_predict = softmax_reg.predict(X_new)
print(y_predict)


#下面为画图
'''
zz1 = y_proba[:, 1].reshape(x0.shape)
zz = y_predict.reshape(x0.shape)

plt.figure(figsize=(10, 4))
plt.plot(X[y==2, 0], X[y==2, 1], "g^", label="Iris virginica")
plt.plot(X[y==1, 0], X[y==1, 1], "bs", label="Iris versicolor")
plt.plot(X[y==0, 0], X[y==0, 1], "yo", label="Iris setosa")

from matplotlib.colors import ListedColormap
custom_cmap = ListedColormap(['#fafab0','#9898ff','#a0faa0'])

plt.contourf(x0, x1, zz, cmap=custom_cmap)
contour = plt.contour(x0, x1, zz1, cmap=plt.cm.brg)
plt.clabel(contour, inline=1, fontsize=12)
plt.xlabel("Petal length", fontsize=14)
plt.ylabel("Petal width", fontsize=14)
plt.legend(loc="center left", fontsize=14)
plt.axis([0, 7, 0, 3.5])
plt.show()'''

```
![在这里插入图片描述](https://img-blog.csdnimg.cn/3dd23f44739e489ea26d2a728bc61690.png?x-oss-process=image/watermark,type_ZHJvaWRzYW5zZmFsbGJhY2s,shadow_50,text_Q1NETiBAd2FubHVOMQ==,size_20,color_FFFFFF,t_70,g_se,x_16)

