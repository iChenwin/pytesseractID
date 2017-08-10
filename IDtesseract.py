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
import json
import re
import numpy as np
import heapq
import threading
import Queue
import psutil
import multiprocessing
import time
import sys

# 单个图片识别item
class ImageRecognizerItem(object):
    # 有点类似其它高级语言的构造函数
    def __init__(self, recognizedText, rect):
        self.rect = rect
        self.recognizedText = recognizedText
        self.dealedText = ""

# 身份证信息类
class IDcardInfo(object):
    # 有点类似其它高级语言的构造函数
    def __init__(self):
        self.IDNumber = ""
        self.name = ""
        self.sex = ""
        self.birthDate = ""
        self.address = ""
        self.issueDate = ""
        self.expiryDate = ""
        self.authority = ""

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__,
                          sort_keys=True, indent=4)

class ThreadRecognize(threading.Thread):
    def __init__(self, queue):
        threading.Thread.__init__(self)
        self.queue = queue

    def run(self):
        while True:
            # check available memory
            virtualMemoryInfo = psutil.virtual_memory()
            availableMemory = virtualMemoryInfo.available
            # print(str(availableMemory/1025/1024)+"M")
            if availableMemory > MEMORY_WARNING:
                args = self.queue.get()
                recognizeImage(*args)
                self.queue.task_done()
            # else:
            #     print("memory warning!")

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

# 图片路径
filePath = '2.jpg'
MEMORY_WARNING = 400*1024*1024  # 200M
CPU_COUNT = multiprocessing.cpu_count() # 线程数
ENABLE_THREAD = True   # 是否开启多线程模式

IDrect = ()
recognizedItems = []
handledTexts = {}

# 使用Tesseract进行文字识别
def recognizeImage(results, cvimage ,rect, language, charWhiteList=None):

    global IDrect

    if IDrect == rect:
        return

    config = "-psm 7"   # single line mode
    if charWhiteList is not None:
        config += " -c tessedit_char_whitelist=" + charWhiteList

    image = Image.fromarray(cvimage)

    result = pytesseract.image_to_string(image, lang=language, config=config)
    string = re.sub("[\s+\.\!\/_,$%^*(+\"\'{}〔〕『』｛｝【】〖〗《》「」〈〉（）()a-zA-Z]+|[+——！，。？、~@#￥%……&*（）“”=:-`′-]+".decode("utf8"), "".decode("utf8"), result)

    if language == "eng" and len(result) == 18:
        handledTexts["IDnumber"] = result
        IDrect = rect
    elif string != "":
        item = ImageRecognizerItem(string, rect)
        results.append(item)

# 省市列表
provinces = [
    "北京",
    "广东",
    "山东",
    "江苏",
    "河南",
    "上海",
    "河北",
    "浙江",
    "香港",
    "陕西",
    "湖南",
    "重庆",
    "福建",
    "天津",
    "云南",
    "四川",
    "广西",
    "安徽",
    "海南",
    "江西",
    "湖北",
    "山西",
    "辽宁",
    "台湾",
    "黑龙江",
    "内蒙古",
    "澳门",
    "贵州",
    "甘肃",
    "青海",
    "新疆",
    "西藏",
    "吉林",
    "宁夏"
]

def handlePersonalInfo():
    for idx, item in enumerate(reversed(recognizedItems)):

        if item.recognizedText.startswith(u"姓名"):
            handledTexts["name"] = item.recognizedText[2:]
        elif item.recognizedText.isdigit() and int(item.recognizedText) > 10000000:
            recognizedItems.remove(item)
        elif item.recognizedText.startswith("19") or item.recognizedText.startswith("20"):
            handledTexts["birthDate"] = item.recognizedText
        elif item.recognizedText.startswith(u"出生"):
            handledTexts["birthDate"] = item.recognizedText[2:]
        elif item.recognizedText.startswith(u"性别"):
            handledTexts["gender"] = item.recognizedText[2:]
        elif item.recognizedText.startswith(u"民族"):
            handledTexts["ethnic"] = item.recognizedText[2:]
        else:
            if item.recognizedText.startswith(u"公民身份号码"):
                if not handledTexts.has_key("IDnumber"):
                    handledTexts["IDnumber"] = item.recognizedText[6:]
                continue
                
            if item.recognizedText.startswith(u"住址"):
                handledTexts["address"] = item.recognizedText[2:]
            else:
                handledTexts["address"] += item.recognizedText[2:]

def main():

    handledTexts["name"] = ""
    handledTexts["birthDate"] = ""
    handledTexts["gender"] = ""
    handledTexts["ethnic"] = ""
    handledTexts["IDnumber"] = ""
    handledTexts["address"] = ""

    # parse command line options
    if len(sys.argv) != 2:
        #print 'Usage: python input_name output_name'
        returnData = {'code':1001, 'data':'无效参数'}
        print json.dumps(returnData)
        exit(1)
    filePath = sys.argv[1]

    start = time.time()

    #print "<----- processing %s ----->" % filePath

    #身份证号码识别，先对图片进行黑白处理，裁剪出身份证号，然后识别
    img = cv2.imread(filePath, 0)
    img = cv2.resize(img, (1200, 900)) 

    # 图片亮度调节
    # imgArr = np.array(img)
    # imgMean = np.mean(img)
    # imgcopy = imgArr - imgMean
    # imgcopy = imgcopy * 2 + imgMean * 3
    # imgcopy = imgcopy / 255

    #二值
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    retval, binaryed = cv2.threshold(img, 110, 255, cv2.THRESH_BINARY);  

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
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (65, 20)) 
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

    queue = Queue.Queue()
    if ENABLE_THREAD:
        for i in range(CPU_COUNT):
            t = ThreadRecognize(queue)
            t.setDaemon(True)
            t.start()

    IDimgs = []
    for idx, IDcnt in enumerate(IDcnts):
        x, y, w, h = cv2.boundingRect(IDcnt)
        rect = (x, y, w, h)
        #裁剪图片，并储存在IDimgs中
        IDimg = img[y: y + h, x: x + w]
        IDimgs.insert(idx, IDimg)

        if ENABLE_THREAD:
            args = (recognizedItems, IDimg, rect, "eng", "0123456789X",)
            queue.put(args)
        else:
            recognizeImage(recognizedItems, IDimg, rect, "eng", "0123456789X")
        # cv2.imshow("IDimg", IDimg)
        # k = cv2.waitKey(0)

    textImgs = []
    for idx, IDcnt in enumerate(contours):
        x, y, w, h = cv2.boundingRect(IDcnt)
        rect = (x, y, w, h)
        if IDrect == rect:
            break

        #裁剪图片，并储存在textImg中
        textImg = binaryed[y: y + h, x: x + w]
        # textImgs.insert(idx, textImg)

        if ENABLE_THREAD:
            args = (recognizedItems, textImg, rect, "chi_sim",)
            queue.put(args)
        else:
            recognizeImage(recognizedItems, textImg, rect, "chi_sim")

        # cv2.imshow("textImg", textImg)
        # k = cv2.waitKey(0)

    queue.join()

    handlePersonalInfo()
    result = json.dumps(handledTexts, default=lambda o: o.__dict__, sort_keys=False, indent=4)
    print json.dumps({'code':1000, 'data':json.loads(result)})
    #print result
    cv2.destroyAllWindows()
    #print "<----- %.1f seconds used ----->" % (time.time() - start)

if __name__ == "__main__":
    main()
    
