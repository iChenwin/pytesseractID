## 使用 pytesseract 识别 18 位身份证号

项目调用 OpenCV 对图片进行预处理，裁剪出包含身份证号码的部分，然后调用 pytesseract 识别出号码。

### 版本
`Python 2.7.13`  

### 依赖库
```
PIL
pytesseract
numpy 1.13.1
```

### 运行

将身份证照片拷贝至项目文件夹下，执行：
`python cvtesserID.py`
