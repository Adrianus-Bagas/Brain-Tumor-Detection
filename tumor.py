import tkinter as tk
from tkinter import *
from tkinter import ttk
from tkinter.scrolledtext import *
from PIL import ImageTk, Image
from tkinter import filedialog
from tensorflow.keras.models import load_model, Model
import numpy as np
import os
import joblib
import cv2
import imutils

window = Tk()
window.title("Tumor Detection")
window.geometry("400x400")


def file():
    currdir = os.getcwd()
    tempdir = filedialog.askopenfilename(parent=window, initialdir=currdir, title='Pilih Gambar')
    inputtxt.insert(END,tempdir)
    img = Image.open(tempdir)
    img = img.resize((100, 100), Image.ANTIALIAS)
    img = ImageTk.PhotoImage(img)
    panel.image = img
    panel.config(image=img)

def deteksi():
    path = inputtxt.get(1.0, "end-1c")
    img = cv2.imread(path)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # threshold the image, then perform a series of erosions +
    # dilations to remove any small regions of noise
    thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)

    # find contours in thresholded image, then grab the largest one
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)

    # find the extreme points
    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])

    # add contour on the image
    img_cnt = cv2.drawContours(img.copy(), [c], -1, (0, 255, 255), 4)

    # add extreme points
    img_pnt = cv2.circle(img_cnt.copy(), extLeft, 8, (0, 0, 255), -1)
    img_pnt = cv2.circle(img_pnt, extRight, 8, (0, 255, 0), -1)
    img_pnt = cv2.circle(img_pnt, extTop, 8, (255, 0, 0), -1)
    img_pnt = cv2.circle(img_pnt, extBot, 8, (255, 255, 0), -1)

    # crop
    ADD_PIXELS = 0
    new_img = img[extTop[1]-ADD_PIXELS:extBot[1]+ADD_PIXELS, extLeft[0]-ADD_PIXELS:extRight[0]+ADD_PIXELS].copy()
    new_img = cv2.resize(
            new_img,
            dsize=(100,100),
            interpolation=cv2.INTER_CUBIC
        )
    X_prep = new_img/255
    model_resnet = load_model('model resnet.h5')
    model_rf = joblib.load('resnet rf.joblib')
    model_feat_resnet = Model(inputs=model_resnet.input,outputs=model_resnet.get_layer('flatten_2').output)
    X_feat_resnet = model_feat_resnet.predict(np.reshape(X_prep, (1,100,100,3)))
    prediction = model_rf.predict(X_feat_resnet)
    if prediction[0]==1:
        outputtxt.insert(END,'Ya')
    else:
        outputtxt.insert(END,'Tidak')

def clear():
    inputtxt.delete("1.0", "end")
    outputtxt.delete("1.0", "end")
    panel.config(image="")

label1 = tk.Label(window, text = "Masukkan Gambar",fg="black")
label1.pack()

b2=Button(text="Load", width=6,command=file)
b2.pack()

inputtxt = tk.Text(window,
                   height = 2,
                   width = 10)
  
inputtxt.pack()

b0=Button(text="Cek", width=6,command=deteksi)
b0.pack()

label2 = tk.Label(window, text = "Apakah ini tumor ?",fg="black")
label2.pack()

outputtxt = tk.Text(window,
                   height = 2,
                   width = 10)
  
outputtxt.pack()

b1=Button(text="Reset", width=6,command=clear)
b1.pack()

global panel
panel = Label(window,image=None)
panel.pack()

window.mainloop()