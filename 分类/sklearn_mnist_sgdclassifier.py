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
plt.imshow(some_digit_image, cmap="binary")
plt.axis("off")
plt.show()
y_train = y_train.astype(np.uint8)
y_test = y_test.astype(np.uint8)
print("x_train is ",y_train[0])

#转变数据集形式
x_train_transed=[]
for index in range(len(x_train)):
    x_train_transed.append(x_train[index].reshape(-1))
    print("疯狂转变中 ",(index/len(x_train))*100,"%")
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
        plt.imshow(some_digit_image, cmap="binary")
        print("image smaple ",i,"predict result ,is 5 :",\
            test_5_model.predict([x_train_transed[i]]))
        plt.axis("off")
        plt.show()
        

