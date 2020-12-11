import cv2
cap = cv2.VideoCapture(r"C:\Users\akito\Downloads\test.raw")

print(cap.get(cv2.CAP_PROP_FRAME_COUNT))
for i in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):
    print("Frame: {}/{}".format(i,cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    ret, frame = cap.read()
    
    cv2.imshow("frame",frame)
    i = i + 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
