import cv2
import numpy as np
import time
import cvPy
from sklearn.decomposition import PCA
from sklearn.linear_model import SGDClassifier
import pickle
from sklearn.metrics import precision_score,recall_score,f1_score
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

YES_SIZE=41+1
NO_SIZE=18+1

def load_img_set(Sum,path):
    cvPy.get_namedWindow("main")
    data=[]
    for i in range(1,Sum):
        now_path=path.format(i)
        img=cv2.imread(now_path,cv2.IMREAD_COLOR)
        # 固定缩放 50*50
        img=cv2.resize(img,(300,300))
        cv2.imshow("main",img)
        cv2.waitKey(10)
        data.append(img)
    return data

def main():
    cvPy.get_namedWindow("me")
    #数据准备  
    yes_data=load_img_set(YES_SIZE,"C:\\Users\\gaowanlu\\Desktop\\MyProject\\A10\\fun\\yes\\yes ({}).jpg") 
    no_data=load_img_set(NO_SIZE,"C:\\Users\\gaowanlu\\Desktop\\MyProject\\A10\\fun\\no\\no ({}).jpg") 
    origin_data=[]
    origin_data.extend(yes_data)
    origin_data.extend(no_data)


    print("Fun YES Sample Count :",len(yes_data)) 
    print("Fun NO Sample Count :",len(no_data))
    #训练集合并
    x_train_transed=[]
    y_train_transed=[]
    for index in range(len(yes_data)):
        x_train_transed.append(yes_data[index].reshape(-1))
        y_train_transed.append(1)
    for index in range(len(no_data)):
        x_train_transed.append(no_data[index].reshape(-1))
        y_train_transed.append(0)


    #PCA降维
    pca = PCA(n_components=34)
    pca.fit(x_train_transed)
    X_reduced=pca.fit_transform(x_train_transed)#x_train_transed#
    # cumsum = np.cumsum(pca.explained_variance_ratio_)#解释方差比
    # dimension = np.argmax(cumsum >= 0.95) + 1
    # print(dimension)#154为最低满足0.95得了 再降维的话将小于0.95
    


    #划分训练集与测试集
    from sklearn.model_selection import train_test_split
    x_train,x_test,y_train,y_test=train_test_split(X_reduced,y_train_transed,test_size=0.4)

    #训练模型
    sgd_clf = SGDClassifier(max_iter=1000, tol=1e-3, random_state=42)
    sgd_clf.fit(x_train,y_train)

    print("正在保存模型")
    with open('./fun.model', 'wb') as fw:
        pickle.dump(sgd_clf, fw)
    
    #模型评估 
    y_train_pred=sgd_clf.predict(x_test)
    precisionScore=precision_score(y_test,y_train_pred)
    recallScore=recall_score(y_test,y_train_pred)
    f1Score=f1_score(y_test,y_train_pred)
    print("精度：",precisionScore,"召回率：",recallScore,"F1 ：",f1Score) 

    # 图像展示 
    for index in range(len(origin_data)):
        #预测img
        test_x=X_reduced[index]
        pred=sgd_clf.predict([test_x])
        print("Result : ",pred)
        img=cvPy.get_putText(origin_data[index],'{}'.format(pred[0]),(200,200),3,(255,0,0),2)
        #图片聚类 k=2 
        gray_img=cvPy.get_Gray(img)
        all=cv2.imread("C:\\Users\\gaowanlu\\Desktop\\MyProject\\A10\\fun\\all.jpg",cv2.IMREAD_COLOR)
        #X = gray_img.reshape(-1, 1)
        # kmeans = KMeans(n_clusters=2, random_state=42).fit(X)

        # segmented_model=kmeans

        # segmented_img = segmented_model.cluster_centers_[kmeans.labels_]
        # segmented_img = segmented_img.astype(np.uint8).reshape(gray_img.shape)
        #segmented_img = cv2.Canny(segmented_img,2,10)
        hsv = cv2.cvtColor(all, cv2.COLOR_BGR2HSV)
        thresh1 = cv2.inRange(hsv,np.array([30, 37, 36]),np.array([77, 255, 255]));
        thresh1=cvPy.get_eroded(thresh1,(3,3),3)
        cv2.imshow("main",img)
        cv2.imshow("me",thresh1)
        cv2.waitKey(500)


def all(index):
    cvPy.get_namedWindow("me")
    cvPy.get_namedWindow("all")
    all=cv2.imread("C:\\Users\\gaowanlu\\Desktop\\MyProject\\A10\\all\\all ({}).jpg".format(index),cv2.IMREAD_COLOR)
    hsv = cv2.cvtColor(all, cv2.COLOR_BGR2HSV)
    thresh1 = cv2.inRange(hsv,np.array([35, 43, 46]),np.array([77, 255, 255]));
    thresh1=cvPy.get_eroded(thresh1,(5,5),5)
    thresh1=cvPy.get_dilate(thresh1,(5,5),3)
    contours=cvPy.get_Contours(thresh1)


    for cnt in contours[0]:
        #cv2.drawContours(all,cnt,-1,(255,0,0),3)
        rect = cv2.minAreaRect(cnt)
        box=cv2.boxPoints(rect)
        box = np.int0(box)
        # print("width",box[1][0])
        # print("height",box[1][1])
        #if(rect[1][0]>1500 and rect[1][1]>1500):
        cv2.drawContours(all,[box],0,(0,0,255),10)


    #all=cvPy.get_drawContours(all,contours)
    # thresh1=cvPy.get_eroded(thresh1,(5,5),10)
    cv2.imshow("me",thresh1)
    cv2.imshow("all",all)
    cv2.waitKey(0)

if __name__ == "__main__":
    main()
    # for i in range(1,36+1):
    #     all(i)