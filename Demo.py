import tensorflow as tf
import cv2
import imutils
from imutils import paths
import numpy as np
import random
from Contrast import Contrast
from Clusterer import Clusterer
import os
from skimage.morphology import skeletonize
from sklearn.metrics import classification_report

CONTRASTER = Contrast()
CLUSTERER = Clusterer()


def white_percent(img):
    # calculated percent of white pixels in the grayscale image
    w, h = img.shape
    total_pixels = w*h
    white_pixels = 0
    for r in img:
        for c in r:
            if c == 255:
                white_pixels += 1
    return white_pixels/total_pixels

# fixes image where number is darker than background in grayscale
def fix_image(img):
    # inversion
    img = cv2.bitwise_not(img)

    # thresholding
    image_bw = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)[1]

    # making mask of a circle
    black = np.zeros((250,250))
    circle_mask = cv2.circle(black, (125, 125), 110, (255, 255, 255), -1) / 255.0

    # applying mask to make everything outside the circle black
    edited_image = image_bw * (circle_mask.astype(image_bw.dtype))
    return edited_image


#**************************************************
# Finding best way to filter image

num_images = 54
i = 0
processed_images = []
image_labels = []

# prepares image paths and randomizes them
image_paths = list(paths.list_images("D:/Data Sets/ColorBlindness/ordered"))
random.shuffle(image_paths)

for imagePath in image_paths:
    image = cv2.imread(imagePath)

    # resize
    image = imutils.resize(image, height=250)

    # contrast
    image = CONTRASTER.apply(image, 60)

    # blurring
    image = cv2.medianBlur(image,15)
    image = cv2.GaussianBlur(image,(3,3),cv2.BORDER_DEFAULT)

    # color clustering
    image = CLUSTERER.apply(image, 5)

    # grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 0.10 - 0.28 should be white
    threshold = 0
    percent_white = white_percent(cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)[1])
    while (not (percent_white > 0.10 and percent_white < 0.28)) and threshold <= 255:
        threshold += 10
        percent_white = white_percent(cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)[1])


    # means that image was not correctly filtered
    if threshold > 255:
        image_bw = fix_image(gray)
    else:
        image_bw = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)[1]

    # blurring
    image_bw = cv2.medianBlur(image_bw,5)
    image_bw = cv2.GaussianBlur(image_bw,(31,31),0)
    image_bw = cv2.threshold(image_bw, 150, 255, cv2.THRESH_BINARY)[1]


    # apply morphology close
    kernel = np.ones((9,9), np.uint8)
    image_bw = cv2.morphologyEx(image_bw, cv2.MORPH_CLOSE, kernel)

    # apply morphology open
    kernel = np.ones((9,9), np.uint8)
    image_bw = cv2.morphologyEx(image_bw, cv2.MORPH_CLOSE, kernel)

    # erosion
    kernel = np.ones((7,7), np.uint8)
    image_bw = cv2.erode(image_bw, kernel, iterations=1)

    # skeletonizing
    image_bw = cv2.threshold(image_bw,0,1,cv2.THRESH_BINARY)[1]
    image_bw = (255*skeletonize(image_bw)).astype(np.uint8)

    # dilating
    kernel = np.ones((21,21), np.uint8)
    image_bw = cv2.morphologyEx(image_bw, cv2.MORPH_DILATE, kernel)


    processed_images.append(imutils.resize(image_bw, height=28))
    image_labels.append(int(os.path.split(imagePath)[0][-1]))


    cv2.imshow("Final", image_bw)
    cv2.waitKey(100)
    cv2.destroyAllWindows()

    i += 1
    print(i)
    if i >= num_images:
        break
model = tf.keras.models.load_model("mnist.h5")

processed_images = np.array(processed_images)
processed_images = processed_images.reshape(processed_images.shape[0], 28, 28, 1)
processed_images = tf.cast(processed_images, tf.float32)

image_labels = np.array(image_labels)
preds = np.argmax(model.predict(processed_images), axis=1)
print(classification_report(image_labels, preds))