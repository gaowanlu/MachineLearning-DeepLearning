#手写数字多分类
from tensorflow import keras
from sklearn.svm import SVC
import numpy as np
import pickle
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
# #转变数据集形式
x_train_transed=[]
for index in range(len(x_train)):
    x_train_transed.append(x_train[index].reshape(-1))
x_test_transed=[]
for index in range(len(x_test)):
    x_test_transed.append(x_test[index].reshape(-1))
print("转变完了")
# svm_clf=SVC()
# print("训练中")
# svm_clf.fit(x_train_transed,y_train)
# print("训练完成")
# print("正在保存模型")
# with open('./multiouput.model', 'wb') as fw:
#     pickle.dump(svm_clf, fw)

print("正在加载模型")
with open('./multiouput.model','rb') as fr:
    model=pickle.load(fr)
    print("使用模型中")
    
result=model.predict([x_train_transed[0],\
    x_train_transed[1],\
    x_train_transed[2],\
    x_train_transed[3],\
    x_train_transed[4]])
print(result)
#实际sklearn训练了45个二元分类器，获得它们对图片的决策分数，然后选择分数最高的
#如何验证? 看score
some_scores=model.decision_function([x_train_transed[0]])
print(some_scores)
#可见5的分数最高
print(model.classes_[np.argmax(some_scores)])

#使用部分数据集
x_train_transed=x_train_transed[:10000]
y_train=y_train[:10000]

#Ovo与OvR
#https://blog.csdn.net/alw_123/article/details/98869193
#OneVsOneClassifier或OneVsRestClassifier类
from sklearn.multiclass import OneVsRestClassifier
ovr_clf=OneVsRestClassifier(SVC())
ovr_clf.fit(x_train_transed,y_train)
print(ovr_clf.predict([x_train_transed[0]]))

#训练SGDClassifier或者RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
sgd_clf = SGDClassifier(max_iter=1000, tol=1e-3, random_state=42)
forest_clf=RandomForestClassifier(random_state=42)
sgd_clf.fit(x_train_transed,y_train)
forest_clf.fit(x_train_transed,y_train)
print(sgd_clf.predict([x_train_transed[0]]))
print(forest_clf.predict([x_train_transed[0]]))

#SGD分类器可以直接将实例分威多个类
print(sgd_clf.decision_function([x_train_transed[0]]))

#交叉验证
from sklearn.model_selection import cross_val_score
result=cross_val_score(sgd_clf,x_train_transed,y_train,cv=3,scoring="accuracy")
print("3折交叉验证 ",result)
#[0.86592681 0.86978698 0.85448545] 都在80%以上 看似还不错
#对输入进行简单缩放
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
x_train_transed=np.array(x_train_transed)
x_train_scaled=scaler.fit_transform(x_train_transed.astype(np.float64))
result=cross_val_score(sgd_clf,x_train_scaled,y_train,cv=3,scoring="accuracy")
print("输入缩放后3折交叉验证 ",result)




from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
#混淆矩阵
y_train_pred=cross_val_predict(sgd_clf,x_train_transed,y_train,cv=3)
conf_mx=confusion_matrix(y_train,y_train_pred)
print(conf_mx)
#使用Matplotlib查看混淆矩阵
plt.matshow(conf_mx, cmap=plt.cm.gray)
plt.show()
row_sums = conf_mx.sum(axis=1, keepdims=True)
norm_conf_mx = conf_mx / row_sums
np.fill_diagonal(norm_conf_mx, 0)#填充对角线为0
plt.matshow(norm_conf_mx, cmap=plt.cm.gray)
plt.show()
#例如(a,b) a 横坐标 b纵坐标 比较白
#则 b被预测为a的数量数较多




#多标签分类器
#形象比喻
#当一张爱丽丝、查理的照片时
#应该输出[1,0,1]
#是爱丽丝、不是鲍勃、是查理
#K近邻
from sklearn.neighbors import KNeighborsClassifier
y_train_large = (y_train >= 7)
y_train_odd = (y_train % 2 == 1)
y_multilabel = np.c_[y_train_large, y_train_odd]
#标签 (a,b) a表示是否>=7 b代表是否为奇数
knn_clf = KNeighborsClassifier()
knn_clf.fit(x_train_transed, y_multilabel)
result=knn_clf.predict([x_train_transed[10]])
print("knn 多标签分类器 ",result)
#模型评测
#平均F1
from sklearn.metrics import f1_score
y_train_knn_pred=cross_val_predict(knn_clf,x_train_transed,y_multilabel,cv=3)
result=f1_score(y_multilabel,y_train_knn_pred,average="macro")
print("knn average f1 value : ",result)



'''多输出分类'''
noise=np.random.randint(0,100,(len(x_train_transed),784),int)
#训练集加噪声
x_train_mod=x_train_transed+noise
noise=np.random.randint(0,100,(len(x_test),784),int)
#测试集加噪声
x_test_mod=x_test_transed+noise
y_train_mod=x_train_transed
y_test_mod=x_test

#有噪声图片作为 x  无噪声图片作为 y
knn_clf.fit(x_train_mod,y_train_mod)

#预测有噪声图片，得到无噪声图片 y
clean_digit=knn_clf.predict([x_test_mod[4]])
print("k紧邻 无噪声图片 :",clean_digit)