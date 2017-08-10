#!/usr/bin/env python
#coding=utf-8

from PIL import Image
import pytesseract
import cv2
import os
import string
import re
import numpy as np
import sys

def main():
    # parse command line options
    if len(sys.argv) != 2:
        print 'Usage: python input_name output_name'
        exit(1)
    filePath = sys.argv[1]

    print "<----- processing %s ----->" % filePath

    #身份证号码识别，先对图片进行黑白处理，裁剪出身份证号，然后识别
    img = cv2.imread(filePath, 0)
    img = cv2.resize(img, (1200, 900)) 

    # 图片亮度调节
    # imgArr = np.array(img)
    # imgMean = np.mean(img)
    # imgcopy = imgArr - imgMean
    # imgcopy = imgcopy * 2 + imgMean * 3
    # imgcopy = imgcopy / 255

    canny = cv2.Canny(img, 60, 300)  
    inverted = cv2.bitwise_not(canny)
    cv2.imshow('Canny', inverted)

    test1 = Image.fromarray(canny)
    test2 = Image.fromarray(inverted)

    result = pytesseract.image_to_string(test1, lang="eng", config="-c tessedit_char_whitelist=0123456789X")
    print result
    print "-------"
    result = pytesseract.image_to_string(test2, lang="eng")
    print result

    k = cv2.waitKey(0)

if __name__ == "__main__":
    main()