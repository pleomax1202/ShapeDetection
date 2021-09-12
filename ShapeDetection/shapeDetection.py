import numpy as np
import argparse
import cv2
import imutils
import os

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', type=str, required=True, help="path to the input image")
ap.add_argument('-o', '--output', type=str, required=True, help="path to the output detected image")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
thresh = cv2.threshold(blurred.copy(), 127, 255, cv2.THRESH_BINARY_INV)[1]
cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

shape = ''
for c in cnts:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.04*peri, True)

    if len(approx) == 3:
        shape = 'Triangle'
    elif len(approx) == 4:
        x, y, w, h = cv2.boundingRect(c)
        ar = w / h
        if 1.05 > ar > 0.95:
            shape = 'Square'  
        else: 
            shape = 'Rectangle'
    else:
        shape = 'Circle'

    M = cv2.moments(c)
    cX = int((M['m10'] / M['m00']))
    cY = int((M['m01'] / M['m00']))
    cv2.putText(image, shape, (cX-10, cY+10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 0)

output_path = os.path.join(args["output"], 'detected.jpg')
cv2.imwrite(output_path, image)