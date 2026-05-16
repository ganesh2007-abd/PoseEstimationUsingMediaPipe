import cv2 as cv
import mediapipe as mp
import time

cap = cv.VideoCapture(0)



mpPose = mp.solutions.pose
pose = mpPose.Pose()

mpDraw = mp.solutions.drawing_utils

while True:
    success,img = cap.read()

    if not success:
        break

    imgRGB = cv.cvtColor(img,cv.COLOR_BGR2RGB)

    results = pose.process(imgRGB)

    
    

    if results.pose_landmarks:
        mpDraw.draw_landmarks(img,results.pose_landmarks,mpPose.POSE_CONNECTIONS)
        


    cv.imshow("Image",img)
    if cv.waitKey(1) == ord('q'):
        break

cap.release()
cv.destroyAllWindows()