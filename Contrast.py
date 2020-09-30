import cv2

class Contrast:
    def apply(self,img,contrast=0):
        buf = img.copy()
        if contrast != 0:
            f = 131*(contrast+127)/(127*(131-contrast))
            alpha_c = f
            gamma_c = 127*(1-f)
            buf = cv2.addWeighted(buf,alpha_c,buf,0,gamma_c)
        return buf