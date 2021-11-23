import cv2
import numpy as np
import pickle
import time
import cvPy.cvPy as cvPy

'''将此文件移动到project目录下 即可运行'''
def main():
    save_count=0
    with open('./facedp_sgd/face.model','rb') as fr:
        face_model=pickle.load(fr)

    cap=cv2.VideoCapture(0)
    #width 640 height  480
    cvPy.get_namedWindow("Video")
    while 1:
        cvPy.get_fps.start()
        success,img=cap.read()
        if(not success):
            break
        #print(img.shape)
        origin_img=img.copy()
        #BGR=cvPy.get_split(origin_img)
        #cv2.imshow("G",BGR[1])
        #img=get_threshold(img,120,255)
        #滑动窗口检测
        result=False
        img=cvPy.get_Gray(img)

        
        x=0
        y=0
        while x <640-260:
            while y <480-307:
                face=cvPy.get_img_ROI(img,x,y,260,307)
                dp_data=face.reshape(1,-1)
                if(face_model.predict(dp_data)):
                    cvPy.get_rect(origin_img,(x,y),(x+260,y+307),(255,0,0),2)
                    result=True
                    x+=260
                    y+=307
                    if(y>=480-307 or x>=640-260):
                        break
                else:
                    y+=10
            x+=10
            y=0

        cvPy.get_fps.end()
        origin_img = cv2.putText(origin_img, 'Result: {} FPS: {}'.format(result,cvPy.get_fps.get_fps()),\
             (0,30), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2)
        cv2.imshow("Video",face)
        cv2.imshow("origin_img",origin_img)
        key=cv2.waitKey(1)
        if(key&0xFF==ord('q')):
            break
        elif(key&0xFF==ord('s')):
            cv2.imwrite('./nosample/{}.jpg'.format(save_count),face, [int( cv2.IMWRITE_JPEG_QUALITY), 95])
            save_count+=1

if __name__ == "__main__":
    main()
