import tkinter as tk
from tkinter import *
from selenium import webdriver
import pandas as pd
import numpy as np
from PIL import Image
import tensorflow as tf
import cv2
import imutils
from imutils import paths
from Contrast import Contrast
from Clusterer import Clusterer
import os
from skimage.morphology import skeletonize
import random
import time

#object instances
CONTRASTER = Contrast()
CLUSTERER = Clusterer()

#methods
def start_test():
    global test_img,i,image_labels,processed_images
    print()
    input_val = []
    processed_images = []
    image_labels = []
    text_label = Label(text='Image',bg='black',fg='white',font=('bold',13))
    text_label.place(x=130,y=200)

    def white_percent(img):
        # calculated percent of white pixels in the grayscale image
        w, h = img.shape
        total_pixels = w * h
        white_pixels = 0
        for r in img:
            for c in r:
                if c == 255:
                    white_pixels += 1
        return white_pixels / total_pixels

    # fixes image where number is darker than background in grayscale
    def fix_image(img):
        # inversion
        img = cv2.bitwise_not(img)

        # thresholding
        image_bw = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)[1]

        # making mask of a circle
        black = np.zeros((250, 250))
        circle_mask = cv2.circle(black, (125, 125), 110, (255, 255, 255), -1) / 255.0

        # applying mask to make everything outside the circle black
        edited_image = image_bw * (circle_mask.astype(image_bw.dtype))
        return edited_image

    def get_num():
        #print(number.get())
        return number.get()

    def next_func():
        global number, i, test_img, val,image_labels, processed_images

        def result():
            global image_labels, processed_images
            y = get_num()
            input_val.append(int(y))
            #print(input_val)
            model = tf.keras.models.load_model("mnist.h5")

            processed_images = np.array(processed_images)
            processed_images = processed_images.reshape(processed_images.shape[0], 28, 28, 1)
            processed_images = tf.cast(processed_images, tf.float32)
            image_labels = np.array(image_labels)
            preds = np.argmax(model.predict(processed_images), axis=1)
            print('Predicted List is:',preds)
            right = 0
            for i in preds:
                if preds[i] == input_val[i]:
                    right += 1
            print('Your score is:',10,' of 12')
            if right < 11:
                print('It is advised to consult a specialist.')

        if i == 0:
            x = get_num()
            input_val.append(int(x))
            #print(input_val)
        else:
            y = get_num()
            input_val.append(int(y))
            #print(input_val)
        i += 1

        if i == 11:
            result = Button(text='Result',bg='white',fg='black',command=result)
            result.place(x=360,y=400)


        image = cv2.imread(image_paths[i])

        # resize
        image = imutils.resize(image, height=250)

        # contrast
        image = CONTRASTER.apply(image, 60)

        # blurring
        image = cv2.medianBlur(image, 15)
        image = cv2.GaussianBlur(image, (3, 3), cv2.BORDER_DEFAULT)

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
        image_bw = cv2.medianBlur(image_bw, 5)
        image_bw = cv2.GaussianBlur(image_bw, (31, 31), 0)
        image_bw = cv2.threshold(image_bw, 150, 255, cv2.THRESH_BINARY)[1]

        # apply morphology close
        kernel = np.ones((9, 9), np.uint8)
        image_bw = cv2.morphologyEx(image_bw, cv2.MORPH_CLOSE, kernel)

        # apply morphology open
        kernel = np.ones((9, 9), np.uint8)
        image_bw = cv2.morphologyEx(image_bw, cv2.MORPH_CLOSE, kernel)

        # erosion
        kernel = np.ones((7, 7), np.uint8)
        image_bw = cv2.erode(image_bw, kernel, iterations=1)

        # skeletonizing
        image_bw = cv2.threshold(image_bw, 0, 1, cv2.THRESH_BINARY)[1]
        image_bw = (255 * skeletonize(image_bw)).astype(np.uint8)

        # dilating
        kernel = np.ones((21, 21), np.uint8)
        image_bw = cv2.morphologyEx(image_bw, cv2.MORPH_DILATE, kernel)

        processed_images.append(imutils.resize(image_bw, height=28))
        image_labels.append(int(os.path.split(image_paths[i])[0][-1]))

        test_img = Image.open(image_paths[i])
        test_img = test_img.resize((int(516 * 2 / 4), int(516 * 2 / 4)), Image.ANTIALIAS)
        test_img = test_img.save('test_img.ppm', 'ppm')
        test_img = PhotoImage(file='test_img.ppm')

        place_img = Label(image=test_img, bg='black')
        place_img.place(x=50, y=235)

        count = Label(text=i + 1, bg='black', fg='white', font=('bold', 13))
        count.place(x=180, y=200)

        enter_num = Label(text='Enter Number:', fg='white', bg='black', font=('bold', 13))
        enter_num.place(x=320, y=300)
        val = IntVar()
        #val.set(100)
        number = Entry(text=val, bg='white', fg='black', justify='center')
        number.place(x=320, y=330)
        next = Button(text='Next', command=next_func, bg='white', fg='black', activebackground='orange')
        next.place(x=360, y=360)
        #input_val.append(int(number.get()))


   ####image labels
    image_labels, processed_images = [],[]

    image_paths = list(paths.list_images("D:/Data Sets/ColorBlindness/test"))
    random.shuffle(image_paths)

    #print(len(image_paths))

    image = cv2.imread(image_paths[i])

    # resize
    image = imutils.resize(image, height=250)

    # contrast
    image = CONTRASTER.apply(image, 60)

    # blurring
    image = cv2.medianBlur(image, 15)
    image = cv2.GaussianBlur(image, (3, 3), cv2.BORDER_DEFAULT)

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
    image_bw = cv2.medianBlur(image_bw, 5)
    image_bw = cv2.GaussianBlur(image_bw, (31, 31), 0)
    image_bw = cv2.threshold(image_bw, 150, 255, cv2.THRESH_BINARY)[1]

    # apply morphology close
    kernel = np.ones((9, 9), np.uint8)
    image_bw = cv2.morphologyEx(image_bw, cv2.MORPH_CLOSE, kernel)

    # apply morphology open
    kernel = np.ones((9, 9), np.uint8)
    image_bw = cv2.morphologyEx(image_bw, cv2.MORPH_CLOSE, kernel)

    # erosion
    kernel = np.ones((7, 7), np.uint8)
    image_bw = cv2.erode(image_bw, kernel, iterations=1)

    # skeletonizing
    image_bw = cv2.threshold(image_bw, 0, 1, cv2.THRESH_BINARY)[1]
    image_bw = (255 * skeletonize(image_bw)).astype(np.uint8)

    # dilating
    kernel = np.ones((21, 21), np.uint8)
    image_bw = cv2.morphologyEx(image_bw, cv2.MORPH_DILATE, kernel)

    processed_images.append(imutils.resize(image_bw, height=28))
    image_labels.append(int(os.path.split(image_paths[i])[0][-1]))

    #cv2.imshow("Final", image_bw)
    #cv2.waitKey(1000)
    #cv2.destroyAllWindows()
    #time.sleep(3)
    test_img = Image.open(image_paths[i])
    test_img = test_img.resize((int(516 * 2 / 4), int(516 * 2 / 4)), Image.ANTIALIAS)
    test_img = test_img.save('test_img.ppm', 'ppm')
    test_img = PhotoImage(file='test_img.ppm')

    place_img = Label(image=test_img,bg='black')
    place_img.place(x=50,y=235)

    count = Label(text=i+1,bg='black',fg='white',font=('bold',13))
    count.place(x=180,y=200)

    enter_num = Label(text='Enter Number:',fg='white',bg='black',font=('bold',13))
    enter_num.place(x=320,y=300)
    val = IntVar()
    number = Entry(text=val,bg='white',fg='black',justify='center')
    number.place(x=320,y=330)
    next = Button(text='Next',command=next_func,bg='white',fg='black',activebackground='orange')
    next.place(x=360,y=360)



def guidelines():
    gl = Toplevel()
    gl.geometry('400x400')
    gl.resizable(0,0)
    gl.configure(background='gray13')
    gl.mainloop()

def selection():
    selection = "You selected the option " + str(var.get())
    print(selection)
    if var.get() == 1:
        male.select()
    else:
        female.select()


def download():
    doc_list = []
    info = driver.find_elements_by_class_name('info-section')
    docs = info
    #print(docs)
    for doc in docs:
        detail = doc.text
        detail = detail.split('\n')
        #print(len(detail)) 6
        #print(detail)
        #print('-------------------------------------------')
        doc_list.append(detail)
    #print(doc_list)
    details = ['Name','Category','Experience','Location','Charge','Ratings']
    df = pd.DataFrame(doc_list,columns=details)
    #print(df)
    df.to_csv('doc_detail.csv')


def search():
    global driver
    name = city_name.get()
    print('Name:',name)
    url = f'https://www.practo.com/{name}/treatment-for-color-blindness'  # city name
    #print(url)
    driver = webdriver.Chrome('chromedriver.exe')
    driver.get(url)
    download_details = Button(doc,text='Download Details',bg='white',fg='black',activebackground='orange',command=download)
    download_details.place(x=160,y=25)
    #driver.quit()

def info():
    about = Toplevel()
    about.geometry('600x300')
    about.resizable(0,0)
    about.configure(background='gray13')
    about.title('About')
    print('Info')
    img = Image.open('bb.png')
    img = img.resize((250, 250), Image.ANTIALIAS)
    img = img.save('colorblindness.png', 'png')
    img = PhotoImage(file='colorblindness.png')

    cb = Label(about,image=img, bg='black')
    cb.place(x=300, y=20)

    text = '\tColor Blindness Toolkit is a portable toolkit \n' \
           'which can be used to help people\n' \
           'suffering from color blindness'
    information = Label(about,text=text,fg='white',bg='gray13')
    information.place(x=0,y=10)

    v = Label(about,text='V', font=('bold', 15), fg='red', bg='gray13')
    v.place(x=95, y=65)

    i = Label(about,text='I', font=('bold', 15), fg='green', bg='gray13')
    i.place(x=115, y=65)

    s = Label(about,text='S', font=('bold', 15), fg='blue', bg='gray13')
    s.place(x=125, y=65)

    i_ = Label(about,text='I', font=('bold', 15), fg='yellow', bg='gray13')
    i_.place(x=145, y=65)

    o = Label(about,text='O', font=('bold', 15), fg='purple', bg='gray13')
    o.place(x=160, y=65)

    n = Label(about,text='N', font=('bold', 15), fg='orange', bg='gray13')
    n.place(x=180, y=65)

    about.mainloop()

def doctor():
    global name,city_name,doc
    doc = Toplevel()
    doc.title('Doctor Details')
    doc.geometry('400x200')
    doc.configure(background='grey13')
    doc.resizable(0,0)

    #city name
    city = Label(doc,text='City',fg='white',bg='gray13')
    city.place(x=25,y=25)

    name = StringVar()
    city_name = Entry(doc,text=name,fg='black',bg='white')
    city_name.place(x=60,y=25)
    search_doc = Button(doc,text='Search Doctor',fg='black',bg='white',activebackground='orange',command=search)
    search_doc.place(x=150,y=150)

    doc.mainloop()


#declarations
i = 0
root = Tk()
root.geometry('800x600')
root.configure(background='black')
root.resizable(0,0)
root.title('Color Blindness Toolkit')

title = Label(text='Color Blindness Toolkit',fg='white',bg='black',font=('bold',16))
title.place(x=300,y=15)

menu = Menu(root)
option = Menu(menu,tearoff=0)
option.add_command(label='Info',command=info)
option.add_separator()
option.add_command(label="Exit", command=root.quit)
menu.add_cascade(label="About", menu=option)


help = Menu(menu,tearoff=0)
help.add_command(label='Doctor info',command=doctor)
menu.add_cascade(label="Help", menu=help)
root.config(menu=menu)

#Doc Image
'''
img = Image.open('colorblindnessinfo.png')
img = img.resize((int(558*2/4), int(682*2/4)), Image.ANTIALIAS)
img = img.save('doc.ppm', 'ppm')
img = PhotoImage(file='doc.ppm')

cbinfo = Label(image=img, bg='black')
cbinfo.place(x=80, y=40)
'''

img1 = Image.open('immm-removebg-preview.png')
img1 = img1.resize((int(582*3/4), int(429*3/4)), Image.ANTIALIAS)
img1 = img1.save('doc.ppm', 'ppm')
img1 = PhotoImage(file='doc.ppm')

doc = Label(image=img1, bg='black')
doc.place(x=380, y=40)

name = Label(text='Name',bg='black',fg='white',font=('bold',13))
name.place(x=20,y=80)
enter_name = Entry(bg='white',fg='black')
enter_name.place(x=80,y=80)

age = Label(text='Age',bg='black',fg='white',font=('bold',13))
age.place(x=20,y=120)
enter_age = Entry(bg='white',fg='black')
enter_age.place(x=80,y=120)

gender = Label(text='Gender',bg='black',fg='white',font=('bold',13))
gender.place(x=20,y=160)
var = IntVar()
#var.set(1)

male = Radiobutton(text='Male',variable=var,value=1,command=selection,bg='black',fg='white',activebackground='orange')
male.place(x=80,y=160)

female = Radiobutton(text='Female',variable=var,value=2,command=selection,bg='black',fg='white',activebackground='orange')
female.place(x=130,y=160)

test = Button(text='Start Test',activebackground='orange',fg='black',bg='white',command=start_test)
test.place(x=220,y=120)

credits = Label(root,bg='black',fg='white',text='©Developed by Bhavishya Pandit',height=3,width=120)
credits.place(x=0,y=550)
root.mainloop()