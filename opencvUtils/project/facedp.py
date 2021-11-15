import cv2
import numpy as np
from sklearn.linear_model import SGDClassifier
import pickle
def main():
    samples=np.array(range(76))
    nosamples=np.array(range(76))
    samples_data=[]
    nosamples_data=[]
    for i in samples:
        img=cv2.imread('C:/Users/gaowanlu/Desktop/MyProject/AI/opencvUtils/sample/{}.jpg'.format(i),cv2.IMREAD_GRAYSCALE)
        b=img.reshape(1,-1)
        #b=(b>100)
        samples_data.append(b)
    for i in nosamples:
        img=cv2.imread('C:/Users/gaowanlu/Desktop/MyProject/AI/opencvUtils/nosample/{}.jpg'.format(i),cv2.IMREAD_GRAYSCALE)
        b=img.reshape(1,-1)
        #b=(b>100)
        nosamples_data.append(b)
    #贴标签
    train_data=np.concatenate((samples_data,nosamples_data))
    samples_tag=(samples<=samples.size)
    nosamples_tag=(nosamples<0)
    train_tag=np.concatenate((samples_tag,nosamples_tag))
    train_data=train_data.reshape(152,-1)
    #训练模型
    sgd_clf = SGDClassifier(max_iter=1000, tol=1e-3, random_state=42)
    sgd_clf.fit(train_data,train_tag)
    print(sgd_clf.predict([train_data[0],train_data[88]]))
    print("正在保存模型")
    with open('../face.model', 'wb') as fw:
        pickle.dump(sgd_clf, fw)













if __name__ == "__main__":
    main()