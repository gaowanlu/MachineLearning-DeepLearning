from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
import pickle

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
print(len(x_train))#60000个数据集
print(len(y_test))#10000个测试集
#print(x_train[0]) #二维List
some_digit = x_train[0]
some_digit_image = some_digit.reshape(28, 28)
#plt.imshow(some_digit_image, cmap="binary")
#plt.axis("off")
#plt.show()
y_train = y_train.astype(np.uint8)
y_test = y_test.astype(np.uint8)
print("x_train is ",y_train[0])

#转变数据集形式
x_train_transed=[]
for index in range(len(x_train)):
    x_train_transed.append(x_train[index].reshape(-1))
    #print("疯狂转变中 ",(index/len(x_train))*100,"%")
print("转变完了")
sgd_clf = SGDClassifier(max_iter=1000, tol=1e-3, random_state=42)

#将数字标签转换为bool型标签,List内item的转换
y_train_5 = (y_train == 5)
print("训练中")
sgd_clf.fit(x_train_transed,y_train_5)
print("训练完了")
print("正在保存模型")
with open('./5_image_test.model', 'wb') as fw:
    pickle.dump(sgd_clf, fw)
print("正在加载模型")
with open('./5_image_test.model','rb') as fr:
    test_5_model=pickle.load(fr)
    print("使用模型中")
    for i in range(10):
        some_digit_image = x_train[i].reshape(28, 28)
        #plt.imshow(some_digit_image, cmap="binary")
        print("image smaple ",i,"predict result ,is 5 :",\
            test_5_model.predict([x_train_transed[i]]))
        #plt.axis("off")
        #plt.show()
        
#交叉验证
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
#StratifiedKFold分层交叉验证法
skfolds =StratifiedKFold(n_splits=3,random_state=42,shuffle=True)
#分成三份 两份作为训练集 一份作为测试集
for train_index ,test_index in skfolds.split(x_train_transed,y_train_5):
    print(len(train_index),len(test_index))
    clone_clf=clone(sgd_clf)
    x_train_folds=np.array(x_train_transed)[train_index]
    y_train_folds=np.array(y_train_5)[train_index]

    x_test_fold=np.array(x_train_transed)[test_index]
    y_test_fold=np.array(y_train_5)[test_index]

    clone_clf.fit(x_train_folds,y_train_folds)
    y_pred=clone_clf.predict(x_test_fold)
    n_correct=sum(y_pred==y_test_fold)
    print(n_correct/len(y_pred))


#使用cross_val_score进行交叉验证
from sklearn.model_selection import cross_val_score
result=cross_val_score(sgd_clf,x_train_transed,y_train_5,cv=3,scoring="accuracy")
print("cross_val_score result ",result)

#准确率通常无法成为分类器的首要性能指标，特别是处理有偏数据集时，
# 大约有10%的图片为5，则随意一张图片不是5，则有90%的概率都是猜正确了


#混淆矩阵
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
y_train_pred=cross_val_predict(sgd_clf,x_train_transed,y_train_5,cv=3)
#cross_val_predict同样执行k-折交叉验证，返回的不是评估分数，而是每个折叠的预测
#这意味着对于每个实例都可以得到一个干净的预测
# (干净的意思是模型预测时使用的数据在其训练期间从未见过)
print(len(y_train_pred))
matrix=confusion_matrix(y_train_5,y_train_pred)
print(matrix)
# [[53892   687]   53892非5预测为非5   687非5被预测为5
#  [ 1891  3530]]   1891 5预测为非5    3530 5预测为5
'''
精度=（TP）/（TP+FP）T原来为正 P预测为正  F原来为负 N预测为负
召回率=(TP)/(TP+FN) 
F1=(2)/((1/精度)+(1/召回率))
'''
from sklearn.metrics import precision_score,recall_score,f1_score
precisionScore=precision_score(y_train_5,y_train_pred)
recallScore=recall_score(y_train_5,y_train_pred)
f1Score=f1_score(y_train_5,y_train_pred)
print("精度：",precisionScore,"召回率：",recallScore,"F1 ：",f1Score)


#如何在精度与召回率之间做抉择
#阈值调整调参优化
y_scores=sgd_clf.decision_function([x_train_transed[0]])
print(y_scores)
#设置阈值为0
threshold=0
temp_pred=(y_scores>threshold)#只留下分数大于阈值的项
print(temp_pred)

#设置阈值为8000
print((y_scores>8000))


#使用cross_val_predict()函数获取训练集中所有实例的分数
y_scores=cross_val_predict(sgd_clf,x_train_transed,y_train_5,cv=3,\
method="decision_function")
#使用precision_recall_curve()函数来计算可能的阈值的精度和召回率

from sklearn.metrics import precision_recall_curve
precisions,recalls,thresholds=precision_recall_curve(y_train_5,y_scores)



def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g-", label="Recall")
    [...] # highlight the threshold and add the legend, axis label, and grid
plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.show()

#假设要将精度设为90%，可以找到精度90%的最低阈值
#阈值越高 召回率越低
threshold_90_percent=thresholds[np.argmax(precisions>=0.9)]
print("保证精度90%以上的最低阈值 ",threshold_90_percent)
#则有了分数阈值，我们也可以使用分数阈值  进行二分类
#Y_train_pred_90=(y_scores>=threshold_90_percent)
#大于阈值可预测为是5


#PR曲线
#画y:精度  x：召回率图像
def plot_precision_vs_recall(precisions, recalls):
    plt.plot(recalls, precisions, "b-", linewidth=2)
    plt.xlabel("Recall", fontsize=16)
    plt.ylabel("Precision", fontsize=16)
    plt.axis([0, 1, 0, 1])
    plt.grid(True)

plot_precision_vs_recall(precisions,recalls)
plt.show()

#ROC曲线(特征曲线)
#真正类率(召回率)和假正类率(FPR)
#FPR=1-TNR
#TNR:正确分类为负类的负类实例比率
from sklearn.metrics import roc_curve
fpr,tpr,thresholds=roc_curve(y_train_5,y_scores)
#画ROC曲线
#y:真正率(召回率) x:假正率
def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--') # Dashed diagonal
    [...] # Add axis labels and grid
plot_roc_curve(fpr, tpr)
#绘制了所有可能阈值的假正率与真正率的关系
plt.show()
#计算ROC曲线下的面积，最理想的ROC曲线下面积为1
#纯随机分类器ROC AUC为0.5
from sklearn.metrics import roc_auc_score
roc_auc_score(y_train_5,y_scores)

#换个随机森林分类器来看看ROC曲线
from sklearn.ensemble import RandomForestClassifier
forest_clf=RandomForestClassifier(random_state=42)
y_probas_forest = cross_val_predict(forest_clf, x_train_transed, y_train_5, cv=3,
method="predict_proba")
#得到的不是分数，是概率，是5的概率

y_scores_forest = y_probas_forest[:, 1] # score = proba of positive class
fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_train_5,y_scores_forest)

plt.plot(fpr, tpr, "b:", label="SGD")
plot_roc_curve(fpr_forest, tpr_forest, "Random Forest")
plt.legend(loc="lower right")
plt.show()
roc_auc_score(y_train_5,y_scores_forest)#计算ROC曲线下面的面积
#随机森林的ROC曲线比SDG分类器ROC曲线，更靠近左上，AUC值更大，效果更好些




