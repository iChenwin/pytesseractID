#!/usr/bin/python
# coding:utf-8

import sys
import threading
import multiprocessing
import Queue
import re
import json
import cv2
import numpy as np
# import os
# import subprocess
import pytesseract
import psutil
# from matplotlib import pyplot as plt
from PIL import Image, ExifTags
from pypinyin import pinyin, lazy_pinyin
import pypinyin
import time

reload(sys)
sys.setdefaultencoding("utf-8")

# 护照中出现的省市列表
passportProvinces = [
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

# 护照中出现的月份英文缩写列表
passportMonthAbbrs = [
    "JAN",
    "FEB",
    "MAR",
    "APR",
    "MAY",
    "JUN",
    "JUL",
    "AUG",
    "SEP",
    "OCT",
    "NOV",
    "DEC"
]

# 根据月份英文缩写获取数字月份
def getMonthNumberStringWithAddr(addr):
    if addr == "JAN":
        return "01"
    elif addr == "FEB":
        return "02"
    elif addr == "MAR":
        return "03"
    elif addr == "APR":
        return "04"
    elif addr == "MAY":
        return "05"
    elif addr == "JUN":
        return "06"
    elif addr == "JUL":
        return "07"
    elif addr == "AUG":
        return "08"
    elif addr == "SEP":
        return "09"
    elif addr == "OCT":
        return "10"
    elif addr == "NOV":
        return "11"
    elif addr == "DEC":
        return "12"
    return ""

def getMidX(rect):
    x0 = rect[0]
    x1 = rect[0] + rect[2]
    return (x0 + x1) * 0.5

def getMidY(rect):
    y0 = rect[1]
    y1 = rect[1] + rect[3]
    return (y0 + y1) * 0.5

# 修正图片旋转
def fixRotation(filePath):
    try:
        image = Image.open(filePath)
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break
        exif = dict(image._getexif().items())

        if exif[orientation] == 3:
            image = image.rotate(180, expand=True)
        elif exif[orientation] == 6:
            image = image.rotate(270, expand=True)
        elif exif[orientation] == 8:
            image = image.rotate(90, expand=True)
        image.save(filePath)
        image.close()

    except (AttributeError, KeyError, IndexError):
        # cases: image don't have getexif
        pass

# 获取暗色在图片中所占百分比
def getDarkColorPercent(image):
    height = np.size(image, 0)
    width = np.size(image, 1)
    imgSize = width * height
    result = cv2.threshold(image, 100, -1, cv2.THRESH_TOZERO)[1]
    nonzero = cv2.countNonZero(result)
    if nonzero > 0:
        return (imgSize - nonzero) / float(imgSize)
    else:
        return 0

# 在图片中画出框
def drawRects(image, rects):
    for rect in rects:
        cv2.rectangle(image, (int(rect[0]), int(rect[1])), (int(
            rect[0] + rect[2]), int(rect[1] + rect[3])), (0, 255, 0), 15, 8, 0)

# 处理成黑白图片以便进行文字识别
def dealImage(image, thresh):
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (2, 1))
    dilate = cv2.dilate(image, kernel)

    gray = cv2.cvtColor(dilate, cv2.COLOR_RGB2GRAY)
    return cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)[1]

# 等比缩放图片
def scaleImage(image, scale):
    height = np.size(image, 0)
    width = np.size(image, 1)
    dstSize = (int(width * scale), int(height * scale))

    return cv2.resize(image, dstSize, None, 0, 0, cv2.INTER_LINEAR)

# 检测可能包含文字的区域
def detectTextRects(image, imageScale):
    # letterBoxes
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    threshold = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY)[1]

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (130, 20))
    result = cv2.dilate((255 - threshold), kernel)

    # // 检索轮廓并返回检测到的轮廓的个数
    contours = cv2.findContours(result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]

    maxValue = 200 * imageScale
    minValue = 40 * imageScale

    boundRect = []
    for points in contours:
        appRect = cv2.boundingRect(points)  # x y w h

        if (appRect[3] > maxValue and appRect[2] > maxValue):
            continue

        if (appRect[3] < minValue or appRect[2] < minValue):
            continue
        appRect = list(appRect)
        appRect[2] += 60 * imageScale
        appRect[3] += 15 * imageScale
        appRect[0] -= 30 * imageScale
        appRect[1] -= 7.5 * imageScale
        boundRect.append(tuple(appRect))
    return boundRect


# 执行文字识别shell并返回结果
# def image_to_string(img, cleanup=True, plus=''):
#     # cleanup为True则识别完成后删除生成的文本文件
#     # plus参数为给tesseract的附加高级参数
#     try:
#         subprocess.check_output('tesseract ' + img + ' ' + img + ' ' + plus, shell=True)  # 生成同名txt文件
#     except subprocess.CalledProcessError as e:
#         return ""
#     text = ''
#     with open(img + '.txt', 'r') as f:
#         text = f.read().strip()
#     if cleanup:
#         os.remove(img + '.txt')
#     return text

# 护照信息类
class PassportInfo(object):
    # 有点类似其它高级语言的构造函数
    def __init__(self):
        self.passportNumber = ""
        self.name = ""
        self.namePinyin = ""
        self.sex = ""
        self.nationality = ""
        self.birthDate = ""
        self.birthPlace = ""
        self.issueDate = ""
        self.issuePlace = ""
        self.expiryDate = ""
        self.authority = ""
        self.authorityEnglish = ""
        self.bearerSignature = ""
        self.firstBooklet = ""
        self.secondBooklet = ""

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__,
                          sort_keys=True, indent=4)

# 单个图片识别item
class ImageRecognizerItem(object):
    # 有点类似其它高级语言的构造函数
    def __init__(self, recognizedText, rect):
        self.rect = rect
        self.recognizedText = recognizedText
        self.dealedText = ""

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

# 判断是否只有中文
def isChinese(string):
    tempString = string.replace(" ", "")
    # tempString = re.sub(ur"[^\u4e00-\u9fa5]", "", unicode(tempString, "utf8"))
    # return len(re.sub(ur"^[\u4e00-\u9fa5]+$", "", unicode(tempString, "utf8"))) == 0
    return len(re.sub(ur"^[\u4e00-\u9fa5]+$", "", tempString)) == 0

# def getChineseString(string):
#     tempString = string.replace(" ", "")
#     tempString = re.sub(ur"[^\u4e00-\u9fa5]", "", unicode(tempString, "utf8"))
#     return tempString.encode("ascii")

# 是否包含数字
def hasNumbers(inputString):
    return bool(re.search(r'\d', inputString))

# 根据正则表达式进行替换，返回替换后的文本是否为空
def replaceWithRegexIsEmpty(regex, string):
    return len(re.sub(regex, "", string)) == 0

# 使用Tesseract进行文字识别
def recognizeImage(results, cvimage ,rect, language, charWhiteList=None):
    config = "-psm 7"   # single line mode
    if charWhiteList is not None:
        config += " -c tessedit_char_whitelist=" + charWhiteList

    image = Image.fromarray(cvimage)

    result = pytesseract.image_to_string(image, lang=language, config=config)

    item = ImageRecognizerItem(result, rect)
    results.append(item)

# 处理ImageRecognizerItem
def handleRecognizedItem(recognizedItem, passportInfo, handledTexts, possibleNames, possibleProvinces, possibleDates):
    recognizedText = recognizedItem.recognizedText.replace("\n", "")
    if (len(recognizedText) > 0):
        dealText = recognizedText.replace("<", "")
        dealText = dealText.replace(" ", "")
        isValid = re.sub("\w", "", dealText) == ""
        string = recognizedText.replace(" ", "")

        # newString = re.sub(ur"^[a-zA-Z0-9\u4e00-\u9fa5,/< ]+$", "", unicode(recognizedText, "utf8"))
        newString = re.sub(ur"^[a-zA-Z0-9\u4e00-\u9fa5,/< ]+$", "", recognizedText)

        # 底部两行
        if (isValid and len(string) == 44 and len(newString) == 0):
            recognizedText = string
            checkDigit = recognizedText[9: 9 + 1]
            if len(re.sub("^[1-9]+$", "", checkDigit)) != 0:
                surname = recognizedText[5:5 + 39]
                arr = filter(None, surname.split("<"))
                if len(arr) == 2:
                    handledTexts["familyName"] = arr[0]
                    handledTexts["givenName"] = arr[1]

                passportInfo.firstBooklet = recognizedText
            else:
                passportNumber = recognizedText[0:0 + 9]
                nationality = recognizedText[10:10 + 3]
                birth = recognizedText[13:13 + 6]
                sex = recognizedText[20:20 + 1]
                expiration = recognizedText[21:21 + 6]
                # personalNumber = recognizedText[28:28+14]

                handledTexts["passportNumber"] = passportNumber
                handledTexts["nationality"] = nationality
                handledTexts["birth"] = birth
                handledTexts["sex"] = sex
                handledTexts["expiration"] = expiration

                passportInfo.passportNumber = passportNumber
                passportInfo.sex = sex
                passportInfo.nationality = nationality
                passportInfo.secondBooklet = recognizedText
        else:
            # detect province
            # 可能是省市：字符串中包含省市的中文或拼音
            for province in passportProvinces:
                provincePinyin = ''.join(lazy_pinyin(unicode(province, 'utf8')))
                # provincePinyin = ''.join(lazy_pinyin(province))
                provincePinyin = provincePinyin.upper()
                string = recognizedText.replace(" ", "")
                if (province in string or provincePinyin in string):
                    recognizedItem.dealedText = province
                    possibleProvinces.append(recognizedItem)

            # detect date
            # 可能是日期：字符串中包含月份缩写
            for monthAddr in passportMonthAbbrs:
                if (monthAddr in recognizedText and hasNumbers(recognizedText)):
                    possibleDates.append(recognizedItem)

            # 可能是姓名：字符串全是中文
            if isChinese(recognizedText):
                # recognizedItem.dealedText = getChineseString(recognizedText).encode("utf8")
                possibleNames.append(recognizedItem)

# 最后处理
# 只针对现版因私普通护照，旧版或其他类型护照的信息位置可能会有所不同
# 护照种类：外交护照、公务护照、普通护照（因公普通护照、因私普通护照）
def handledTextsForPassport(passportInfo, handledTexts, possibleNames, possibleProvinces, possibleDates):
    # find name
    # 条件：字符串拼音是在booklet中检测出的中文姓名拼音
    if handledTexts.has_key("familyName") and handledTexts.has_key("givenName"):
        fullnamePinyin = handledTexts["familyName"] + handledTexts["givenName"]
        fullnamePinyin = fullnamePinyin.upper()
        for item in possibleNames:
            name = item.recognizedText.replace(" ", "")
            # namePinyin = ''.join(lazy_pinyin(unicode(name, 'utf-8')))
            namePinyin = ''.join(lazy_pinyin(name))
            namePinyin = namePinyin.upper()
            if namePinyin == fullnamePinyin:
                passportInfo.name = name
                passportInfo.namePinyin = namePinyin

    # handle province
    # 条件：因为只会出现两个省市，并且上面的是出生地点，下面是签发地点
    if len(possibleProvinces) == 2:
        item0 = possibleProvinces[0]
        item1 = possibleProvinces[1]
        if (getMidY(item0.rect) > getMidY(item1.rect)):
            passportInfo.issuePlace = item0.dealedText
            passportInfo.birthPlace = item1.dealedText

        else:
            passportInfo.issuePlace = item1.dealedText
            passportInfo.birthPlace = item0.dealedText

    # issue and expiry date
    # 条件：27MAY1993类型的日期只有一个（出生日期）；122月FEB2016有两个，上面是签发日期，下面是有效期至，如果只检测到一个此类型日期，根据booklet中检测出的两位年的日期920527，与之匹配检测出是签发日期还是有效期至
    issueOrExpiry = []
    births = []
    for item in possibleDates:
        date = item.recognizedText
        date = date.replace(" ", "")
        date = date.replace("/", "")

        # 27MAY1993
        if replaceWithRegexIsEmpty("^\d{2}[A-Za-z]{3}\d{4}$", date):
            births.append(date)
        # 122月FEB2016
        elif replaceWithRegexIsEmpty(u"^\d{3,4}月{1}[A-Za-z]{3}\d{4}$", date):
            issueOrExpiry.append(date)

    if len(births) == 1 and handledTexts.has_key("birth"):
        date = births[0]
        bookletDate = handledTexts["birth"]

        birthYear = date[5:5 + 4]
        birthMonth = getMonthNumberStringWithAddr(date[2:2 + 3])
        if birthMonth != "":
            birthDay = date[0:0 + 2]

            # 与booklet上的日期比对
            if birthDay == bookletDate[4:4 + 2] and birthMonth == bookletDate[2:2 + 2] and birthYear.endswith(bookletDate[0:0 + 2]):
                passportInfo.birthDate = birthYear + " " + birthMonth + " " + birthDay

    if len(issueOrExpiry) > 0 and handledTexts.has_key("expiration"):
        bookletDate = handledTexts["expiration"]

        date0 = issueOrExpiry[0]
        day0 = date0[0:0 + 2]
        year0 = date0[len(date0) - 4:len(date0) - 4 + 4]
        monthAddr0 = date0[len(date0) - 7:len(date0) - 7 + 3]
        month0 = getMonthNumberStringWithAddr(monthAddr0)

        existExpiryDate = False
        # 与booklet上的日期比对
        if day0 == bookletDate[4:4 + 2] and month0 == bookletDate[2:2 + 2] and year0.endswith(bookletDate[0:0 + 2]):
            passportInfo.expiryDate = year0 + " " + month0 + " " + day0
            existExpiryDate = True

        if len(issueOrExpiry) == 2:
            date1 = issueOrExpiry[1]
            day1 = date1[0:0 + 2]
            year1 = date1[len(date1) - 4:len(date1) - 4 + 4]
            monthAddr1 = date1[len(date1) - 7:len(date1) - 7 + 3]
            month1 = getMonthNumberStringWithAddr(monthAddr1)

            if not existExpiryDate:
                # 与booklet上的日期比对
                if day1[4:4 + 2] and month1[2:2 + 2] and year1.endswith(bookletDate[0:0 + 2]):
                    passportInfo.expiryDate = year1 + " " + month1 + " " + day1
                    passportInfo.issueDate = year0 + " " + month0 + " " + day0

                    existExpiryDate = True
            else:
                passportInfo.issueDate = year1 + " " + month1 + " " + day1

# 图片路径
filePath = '666.jpg'
IMAGE_SCALE = 0.7
MEMORY_WARNING = 400*1024*1024  # 200M
ENABLE_THREAD = True 
def main():
    if len(sys.argv) != 2:
        print 'Usage: python aruba.py image_name'
        exit(1)
    filePath = sys.argv[1]

    passportInfo = PassportInfo()
    recognizedItems = []
    handledTexts = {}
    possibleNames = []
    possibleProvinces = []
    possibleDates = []

    threads = []

    IMAGE_SCALE = 0.7
    MEMORY_WARNING = 400*1024*1024  # 200M
    CPU_COUNT = multiprocessing.cpu_count() # 线程数

    #print '------------'+str(CPU_COUNT)+'----------------'

    ENABLE_THREAD = True   # 是否开启多线程模式

    # 修正图片旋转，有时候手机拍出的照片会出现旋转的情况
    fixRotation(filePath)

    # 读取图片
    img = cv2.imread(filePath, 1)
    height = np.size(img, 0)
    width = np.size(img, 1)
    scale = 4000.0 * IMAGE_SCALE / width * 1.0

    # 拉伸图片到宽度为4000*IMAGE_SCALE
    img = scaleImage(img, scale)

    # 处理图片，以便使用Tesseract进行识别
    dealedImg = dealImage(img, 95)

    # 获取可能包含文字的区域
    rects = detectTextRects(img, IMAGE_SCALE)

    # 测试，画出框
    drawRects(img, rects)
    cv2.imwrite('pas.jpg', img)

    start_time = time.time()


    queue = Queue.Queue()
    if ENABLE_THREAD:
        for i in range(CPU_COUNT):
            t = ThreadRecognize(queue)
            t.setDaemon(True)
            t.start()

    for rect in rects:
        x = int(rect[0])
        y = int(rect[1])
        w = int(rect[2])
        h = int(rect[3])

        # 根据长宽过滤不太可能包含文字的图片
        if (((w > 50 * IMAGE_SCALE and w < 2000 * IMAGE_SCALE) or w > 2500 * IMAGE_SCALE) and (w > h)):
            crop_img = dealedImg[y:y + h, x:x + w]

            darkColorPercent = getDarkColorPercent(crop_img)

            # 根据图片中包含的黑色百分比过滤不太可能包含文字的图片
            if (darkColorPercent > 0.04 and darkColorPercent < 0.35):

                # result = ""

                # 长度很长的很可能就是booklets
                if w > 2500 * IMAGE_SCALE:
                    if ENABLE_THREAD:
                        args = (recognizedItems, crop_img, rect, "eng", "0123456789ABCDEFGHIJKMLNOPQRSTUVWXYZ\<",)
                        # thread = threading.Thread(target=recognizeImage, args=(queue, recognizedItems, crop_img, rect, "eng", "0123456789ABCDEFGHIJKMLNOPQRSTUVWXYZ\<",))
                        # threads.append(thread)
                        queue.put(args)
                    else:
                        recognizeImage(recognizedItems, crop_img, rect, "eng", "0123456789ABCDEFGHIJKMLNOPQRSTUVWXYZ\<")
                else:
                    if ENABLE_THREAD:
                        args = (recognizedItems, crop_img, rect, "eng+chi_sim",)
                        # thread = threading.Thread(target=recognizeImage, args=(queue, recognizedItems, crop_img, rect, "eng+chi_sim",))
                        # threads.append(thread)
                        queue.put(args)
                    else:
                        recognizeImage(recognizedItems, crop_img, rect, "eng+chi_sim")
                

    # if ENABLE_THREAD:
    #     for t in threads:
    #         t.setDaemon(True)
    #         t.start()
    #     # t.join()
    #     # for t in threads:
    #     #     t.join()
    #     queue.join()

    queue.join()

    for item in recognizedItems:
        # 对每个识别出的文字进行处理
        handleRecognizedItem(item, passportInfo, handledTexts, possibleNames, possibleProvinces, possibleDates)

    # # 对收集到的信息进行最后处理
    handledTextsForPassport(passportInfo, handledTexts, possibleNames, possibleProvinces, possibleDates)

    print("--- %s seconds ---" % (time.time() - start_time))

    result = passportInfo.toJSON()
    #print(json.dumps(result))
    print(result)

if __name__ == "__main__":
    main()