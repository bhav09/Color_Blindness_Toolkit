import cv2
import numpy as np

class Clusterer:
    def apply(self,img,K):
        image = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        vectorized = image.reshape((-1,3))
        vectorized = np.float32(vectorized)
        criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,10,1.0)
        attempts = 10
        ret,label,center = cv2.kmeans(vectorized,K,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)
        res = center[label.flatten()]
        resultant_image = res.reshape((image.shape))
        return resultant_image
    