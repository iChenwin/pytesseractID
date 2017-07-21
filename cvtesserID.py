#!/usr/bin/env python
#coding=utf-8

""" Use pytesseract(Google OCR library) recognize ID number.
    使用 pytesseract 识别 18 位身份证号。
"""

from PIL import Image
import pytesseract
import cv2
import os
import string
import re
import numpy as np
import heapq

def main():
    # for i in range(1, 16):
    #     tesseractID("./" + str(i) + ".jpg")

    # 识别当前目录下所有.jpg图片中的身份证号
    imgFiles = []
    for idx, filename in enumerate(os.listdir(".")):
        if filename.endswith(".jpg"):
            imgFiles.insert(idx, filename)
    for img in imgFiles:
        tesseractID("./" + img)

#身份证号码识别，先对图片进行黑白处理，裁剪出身份证号，然后识别
def tesseractID(path):
    img = cv2.imread(path, 0)
    img = cv2.resize(img, (1200, 900)) 

    #二值
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    retval, binaryed = cv2.threshold(img, 80, 255, cv2.THRESH_BINARY);  

    #显示处理后图片，调试用
    # cv2.imshow("Binary", binaryed)
    # k = cv2.waitKey(0)

    #闭运算  
    # closed = cv2.morphologyEx(binaryed, cv2.MORPH_CLOSE, kernel)  
    # cv2.imshow("Close",closed)
    # k = cv2.waitKey(0)


    #开运算  
    # opened = cv2.morphologyEx(binaryed, cv2.MORPH_OPEN, kernel)  
    # cv2.imshow("Open", opened)
    # k = cv2.waitKey(0)

    #腐蚀图像
    # dilated = cv2.dilate(binaryed, kernel) 
    # cv2.imshow("dilate", dilated)
    # k = cv2.waitKey(0)

    #膨胀图像，使身份证号连成一整块，方便裁剪
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 20)) 
    eroded = cv2.erode(binaryed, kernel) 

    # cv2.imshow("cannyed", eroded)
    # k = cv2.waitKey(0)

    #黑白反色，将字转为白色，为下一步框选做准备
    inverted = cv2.bitwise_not(eroded)

    # cv2.imshow("inverted", inverted)
    # k = cv2.waitKey(0)
    
    #框选出前景中，识别出的文本块
    contours, hierarchy = cv2.findContours(inverted, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)  
    
    #在所有文本框中挑出最长的三个框，身份证号应该在其中
    IDcnts = findIDcnt(contours)

    #画框
    # cv2.drawContours(img, IDcnts, -1, (255,0,0), 3)
    # cv2.imshow("img", img)
    # k = cv2.waitKey(0)

    IDimgs = []
    for idx, IDcnt in enumerate(IDcnts):
        x, y, w, h = cv2.boundingRect(IDcnt)
        #裁剪图片，并储存在IDimgs中
        IDimg = img[y: y + h, x: x + w]
        IDimgs.insert(idx, IDimg)

        # cv2.imshow("IDimg", IDimg)
        # k = cv2.waitKey(0)

    #将三张可能的框出的图片丢给tesseract识别，得到身份证
    IDstring = tesseractImg(IDimgs)
    print(path + ": " + IDstring)

    cv2.destroyAllWindows()

#在所有的框中挑出三个最宽的矩形框
def findIDcnt(countours):
    #保存所有框的宽度
    widths = []
    for idx, cnt in enumerate(countours):
        x, y, width, height = cv2.boundingRect(cnt)
        widths.insert(idx, width)
    
    #挑出宽度前三的三个宽度
    IDList = heapq.nlargest(3, widths)
    #根据这三个宽度，找出对应的那三个矩形框
    IDcnts = []
    for idx, item in enumerate(IDList):
        index = widths.index(item)
        IDcnts.insert(idx, countours[index])
    # print IDcnts

    return IDcnts

#tesseract识别
def tesseractImg(imgs):
    for img in imgs:
        result = pytesseract.image_to_string(Image.fromarray(img), lang='chi_sim', config="-c tessedit_char_whitelist=0123456789X")
        # IDstring = re.sub("[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）“”=:-`′-]+".decode("utf8"), "".decode("utf8"), result)    #出去字符串中的标点符号

        if (len(result) == 18):
            return result

    return "ID not found!"

if __name__ == "__main__":
    main()