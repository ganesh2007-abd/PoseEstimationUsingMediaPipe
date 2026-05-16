import cv2 as cv
import mediapipe as mp
import time

cap = cv.VideoCapture(r"C:\Users\HP\Downloads\8402080-hd_1080_1920_30fps.mp4")

ptime = 0

mpPose = mp.solutions.pose
pose = mpPose.Pose()

mpDraw = mp.solutions.drawing_utils

while True:
    success,img = cap.read()

    if not success:
        break

    img = cv.resize(img,None,fx=0.5,fy=0.5,interpolation=cv.INTER_LINEAR)

    imgRGB = cv.cvtColor(img,cv.COLOR_BGR2RGB)

    results = pose.process(imgRGB)

    ctime = time.time()
    fps = 1/(ctime - ptime)
    ptime = ctime

    cv.putText(img,str(int(fps)),(50,70),cv.FONT_HERSHEY_COMPLEX,3,(255,0,255),2)

    if results.pose_landmarks:
        mpDraw.draw_landmarks(img,results.pose_landmarks,mpPose.POSE_CONNECTIONS)
        for id,lm in enumerate(results.pose_landmarks.landmark):
            h,w,c = img.shape
            cx,cy = int(lm.x*w),int(lm.y*h)
            print(id , lm)
            if (id==12):
                cv.circle(img,(cx,cy),10,(255,0,255),4)
        


    cv.imshow("Image",img)
    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()