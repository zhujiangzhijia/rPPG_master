import cv2
import numpy as np

if __name__ == "__main__" :
    im = cv2.imread('img/lenna.png')

    # 任意の描画したいポリゴンの頂点を与える
    contours = np.array(
        [
            [200, 0],
            [260, 160],
            [400, 160],
            [300, 240],
            [400, 400],
            [200, 320],
        ]
    )

    cv2.fillConvexPoly(im, points =contours, color=(255, 25, 255))
    
    cv2.imshow('result', im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()