import matplotlib.pyplot as plt
import cv2
import numpy as np
# OpenCVの設定

filepath = r"C:\Users\akito\source\WebcamRecorder\output\UmcompressedVideo_2.avi"
cap = cv2.VideoCapture(filepath)
fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
video = cv2.VideoWriter('out2.mp4', fourcc, 20.0, (300, 300))

theta = np.linspace(0, 2*np.pi, 201)
delta_theta = np.linspace(0, 2*np.pi, 201)
b = 1.0
muki = True

for i in delta_theta:
    ret, frame = cap.read()
    x = np.cos(theta)
    y1 = np.sin(theta + i)
    y2 = np.sin(b*theta)
    y3 = np.sin(2*theta + i)

    # 描画
    fig = plt.figure(figsize=(3,3))
    plt.plot(x, y1, 'b', x, y2, 'g', x, y3, 'r')
    fig.canvas.draw()

    # VideoWriterへ書き込み
    image_array = np.array(fig.canvas.renderer.buffer_rgba())
    # image_array = np.array(fig.canvas.renderer._renderer) # matplotlibが3.1より前の場合
    im = cv2.cvtColor(image_array, cv2.COLOR_RGBA2BGR)
    concat_frame = cv2.hconcat([im,frame])
    cv2.imshow("Frame2", concat_frame)


    video.write(im)
    plt.close()

    if muki is True:
        b = b + 0.05
        if b > 6:
            muki = False
    else:
        b = b - 0.05
    cv2.waitKey(25)

cap.release()
video.release()
