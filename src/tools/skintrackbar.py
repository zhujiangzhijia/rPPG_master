import cv2
import numpy as np
from ..roi_detection.landmark_extractor import *


def nothing(x):
    pass

def trackbar(df,path,filename):
    # Import landmark 
    pix_x_frames = df.loc[:, df.columns.str.contains('x_')].values.astype(np.int)
    pix_y_frames = df.loc[:, df.columns.str.contains('y_')].values.astype(np.int)

    cap = cv2.VideoCapture(path)
    cv2.namedWindow('marking')
    cv2.createTrackbar('Y Lower','marking',0,255,nothing)
    cv2.createTrackbar('Y Higher','marking',255,255,nothing)
    cv2.createTrackbar('Cb Lower','marking',138,255,nothing)
    cv2.createTrackbar('Cb Higher','marking',173,255,nothing)
    cv2.createTrackbar('Cr Lower','marking',67,255,nothing)
    cv2.createTrackbar('Cr Higher','marking',133,255,nothing)
    i = 0
    while True:
        pix_x = pix_x_frames[i,:].reshape(-1, 1)
        pix_y = pix_y_frames[i,:].reshape(-1, 1)
        _,frame = cap.read()

        # FaceMask by features point
        mask = RoIDetection(frame,pix_x,pix_y)
        img = cv2.bitwise_and(frame, frame, mask=mask)

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        hL = cv2.getTrackbarPos('Y Lower','marking')
        hH = cv2.getTrackbarPos('Y Higher','marking')
        sL = cv2.getTrackbarPos('Cb Lower','marking')
        sH = cv2.getTrackbarPos('Cb Higher','marking')
        vL = cv2.getTrackbarPos('Cr Lower','marking')
        vH = cv2.getTrackbarPos('Cr Higher','marking')
        YCbCr_MIN = np.array([hL, sL, vL])
        YCbCr_MAX = np.array([hH, sH, vH])
        
        LowerRegion = np.array([hL,sL,vL],np.uint8)
        upperRegion = np.array([hH,sH,vH],np.uint8)
        redObject = cv2.inRange(hsv,LowerRegion,upperRegion)
        res1=cv2.bitwise_and(img, img, mask = redObject)
        
        roi_pixel = np.all(img != 0, axis=2)
        skin_pixel = np.all(res1 != 0, axis=2)
        text = "{:.1f} %".format(100*np.sum(skin_pixel)/np.sum(roi_pixel))
        cv2.putText(res1,text, (0, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow(filename,res1)
        
        i = i+1

        if cv2.waitKey(10) & 0xFF == ord('q'):
            np.save(filename, [YCbCr_MIN,YCbCr_MAX])
            print(filename)
            cv2.destroyAllWindows()
            break
