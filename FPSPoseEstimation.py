import cv2 as cv
import mediapipe as mp
import time

cap = cv.VideoCapture(0)

ptime = 0

mpPose = mp.solutions.pose
pose = mpPose.Pose()

mpDraw = mp.solutions.drawing_utils

while True:
    success,img = cap.read()

    if not success:
        break

    imgRGB = cv.cvtColor(img,cv.COLOR_BGR2RGB)

    results = pose.process(imgRGB)

    ctime = time.time()
    fps = 1/(ctime - ptime)
    ptime = ctime

    cv.putText(img,str(int(fps)),(50,70),cv.FONT_HERSHEY_COMPLEX,3,(255,0,255),2)

    if results.pose_landmarks:
        mpDraw.draw_landmarks(img,results.pose_landmarks,mpPose.POSE_CONNECTIONS)
        # print(results)


    cv.imshow("Image",img)
    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()