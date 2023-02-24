import os
import sys

import cv2
import numpy as np
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QMainWindow, QApplication, QGraphicsScene, QFileDialog, QWidget
from main import Ui_MainWindow
from widget3 import Ui_Form
from widget1 import Ui_Form1
from widget2 import Ui_Form2
from srcWindow import Ui_srcForm
from suggestions import Ui_Suggestions

envpath = r'D:\Anaconda\envs\sym-qt\Library\plugins\platforms'
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = envpath


# 主窗口类
class MainWindow(QMainWindow, Ui_MainWindow):
    # 类初始化
    def __init__(self):
        # 调用父类的初始化
        super(MainWindow, self).__init__()
        # 窗口界面初始化
        self.setupUi(self)
        # 当前打开的图片文件名，初始化默认空
        self.__Name = None
        self.__fileName = None
        self.__interName = None
        self.__objectName = None
        self.__srcImageRGB = None
        self.__outImageRGB = None
        self.__interpret = None

        # 文件菜单
        self.actionopenfile.triggered.connect(self.__openFileAndShowImage)
        # 用户意见
        self.actionYour_suggestions.triggered.connect(self.__suggestions)


    # 打开文件并在主窗口中显示打开的图像
    def __openFileAndShowImage(self):
        # 打开文件选择窗口
        __fileName, _ = QFileDialog.getOpenFileName(self, '选择图片', '.', 'Image Files(*.png *.jpeg *.jpg *.bmp)')
        __Name = os.path.basename(__fileName)
        __interName = "D:/interpretation/" + __Name
        __objectName = "D:/object/" + __Name
        # 文件存在
        if __fileName and os.path.exists(__fileName) and os.path.exists(__interName) and os.path.exists(__objectName):
            # 设置打开的文件名属性
            self.__Name = __Name
            self.__fileName = __fileName
            self.__interName = __interName
            self.__objectName = __objectName
            # 转换颜色空间，cv2默认打开BGR空间，Qt界面显示需要RGB空间，所以就统一到RGB吧
            __bgrImg = cv2.imread(self.__fileName)
            __interImg = cv2.imread(self.__interName)
            __objectImg = cv2.imread(self.__objectName)
            # 设置初始化数据
            self.__srcImageRGB = cv2.cvtColor(__bgrImg, cv2.COLOR_BGR2RGB)
            self.__interImageRGB = cv2.cvtColor(__interImg, cv2.COLOR_BGR2RGB)
            self.__objectImageRGB = cv2.cvtColor(__objectImg, cv2.COLOR_BGR2RGB)

            # 在窗口中中间QGraphicsView区域处理
            self.__projection(self.__srcImageRGB)
            # 在窗口左侧QGraphicsView区域显示图片
            self.__drawImage(self.objectView, self.__objectImageRGB)
            # 在窗口中右侧QGraphicsView区域显示图片
            self.__projection2(self.__interImageRGB)

    def __suggestions(self):
        self.form5 = QWidget()
        self.ui5 = Ui_Suggestions()
        self.ui5.setupUi(self.form5)
        self.form5.show()


    # 在窗口中指定的QGraphicsView区域（左或右）显示指定类型（rgb、灰度、二值）的图像
    def __drawImage(self, location, img):
        # RBG图
        if len(img.shape) > 2:
            # 获取行（高度）、列（宽度）、通道数
            __height, __width, __channel = img.shape
            # 转换为QImage对象，注意第四、五个参数
            __qImg = QImage(img, __width, __height, __width * __channel, QImage.Format_RGB888)
        # 灰度图、二值图
        else:
            # 获取行（高度）、列（宽度）、通道数
            __height, __width = img.shape
            # 转换为QImage对象，注意第四、五个参数
            __qImg = QImage(img, __width, __height, __width, QImage.Format_Indexed8)

        # 创建QPixmap对象
        __qPixmap = QPixmap.fromImage(__qImg)
        # 创建显示容器QGraphicsScene对象
        __scene = QGraphicsScene()
        # 填充QGraphicsScene对象
        __scene.addPixmap(__qPixmap)
        # 将QGraphicsScene对象设置到QGraphicsView区域实现图片显示
        location.setScene(__scene)

    def __projection(self, img):
        src = img[:, :, [2, 1, 0]]
        src = src.copy()
        img = src
        # save_path = "D:/Docu/junzheng/data/process"
        for i in range(0, 5):
            img = cv2.medianBlur(img, 5)
        # 输出图像尺寸和通道信息
        sp = img.shape
        print("图像信息：", sp)
        sz1 = sp[0]  # height(rows) of image
        sz2 = sp[1]  # width(columns) of image
        sz3 = sp[2]  # the pixels value is made up of three primary colors
        print('width: %d \n height: %d \n number: %d' % (sz2, sz1, sz3))
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        retval, threshold_img = cv2.threshold(gray_img, 120, 255, cv2.THRESH_BINARY)

        # 垂直投影分割图像
        gray_value_y = []
        for i in range(sz2):
            white_value = 0
            for j in range(sz1):
                if threshold_img[j, i] == 255:
                    white_value += 1
            gray_value_y.append(white_value)
        print("", gray_value_y)
        # 创建图像显示垂直投影分割图像结果
        veri_projection_img = np.zeros((sp[0], sp[1], 1), np.uint8)
        for i in range(sz2):
            for j in range(gray_value_y[i]):
                veri_projection_img[j, i] = 255
        # cv2.imwrite(os.path.join(save_path, "veri_projection_img.jpg"), veri_projection_img)
        text_rect = []

        # 根据垂直投影分割识别列
        inline_y = 0
        start_y = 0
        text_rect_y = []
        for i in range(len(gray_value_y)):
            if inline_y == 0 and gray_value_y[i] > 30:
                inline_y = 1
                start_y = i
            elif inline_y == 1 and gray_value_y[i] < 30 and (i - start_y) > 5:
                inline_y = 0
                if i - start_y > 10:
                    rect = [start_y, i + 1]
                    text_rect_y.append(rect)
            elif inline_y == 1 and i == len(gray_value_y) - 1:
                inline_y = 0
                if i - start_y > 10:
                    rect = [start_y - 1, i + 1]
                    text_rect_y.append(rect)
        text_rect_y.reverse()
        print("分列区域，每列数据起始位置Y：", text_rect_y)
        # 每列数据分段
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 3))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilate_img = cv2.dilate(threshold_img, kernel)
        # cv2.imwrite(os.path.join(save_path, "dilate_img.jpg"), dilate_img)
        for rect in text_rect_y:
            cropImg = dilate_img[0:sp[0], rect[0]:rect[1]]  # 裁剪图像y-start:y-end,x-start:x-end
            sp_x = cropImg.shape
            # 水平投影分割图像
            gray_value_x = []
            for i in range(sp_x[0]):
                white_value = 0
                for j in range(sp_x[1]):
                    if cropImg[i, j] == 255:
                        white_value += 1
                gray_value_x.append(white_value)
            # 创建图像显示水平投影分割图像结果
            hori_projection_img = np.zeros((sp_x[0], sp_x[1], 1), np.uint8)
            for i in range(sp_x[0]):
                for j in range(gray_value_x[i]):
                    veri_projection_img[i, j] = 255
            # cv2.imwrite(os.path.join(save_path, "hori.jpg"), hori_projection_img)
            # 根据水平投影分割识别行
            inline_x = 0
            start_x = 0
            text_rect_x = []
            ind = 0
            for i in range(len(gray_value_x)):
                ind += 1
                if inline_x == 0 and gray_value_x[i] > 10:
                    inline_x = 1
                    start_x = i
                elif inline_x == 1 and gray_value_x[i] < 2 and (i - start_x) > 10:
                    inline_x = 0
                    if i - start_x > 10:
                        rect_x = [start_x - 1, i + 1]
                        text_rect_x.append(rect_x)
                        text_rect.append([start_x - 1, i + 1, rect[0], rect[1]])
                        # cropImg_rect = threshold_img[start_x - 1:i + 1, rect[0]:rect[1]]  # 裁剪二值化图像
                        # crop_img = img[start_x - 1:i + 1, rect[0]:rect[1]] #裁剪原图像
                        # cv2.imshow("cropImg_rect", cropImg_rect)
                        # cv2.imwrite(os.path.join(crop_path,str(ind)+".jpg"),crop_img)
                        # break
                # break

        sum = 0
        dis = 0
        cnt = 0

        # 获得平均高度
        for i in range(np.shape(text_rect)[0]):
            sum += text_rect[i][1] - text_rect[i][0]
        height = sum / np.shape(text_rect)[0]
        print("height:", height)

        # 获取平均间距
        for i in range(np.shape(text_rect)[0] - 1):
            if text_rect[i][2] == text_rect[i + 1][2]:
                dis += text_rect[i + 1][0] - text_rect[i][1]
                cnt += 1
                i += 1
            else:
                i += 1
        dis = dis / cnt
        print("distance:", dis)

        h_stand = 0.7 * height  # 小于认为存在分离问题
        dis_stand = dis  # 判断向上或向下合并（默认向下）
        print("h_stand:", h_stand)
        print("dis_stand:", dis_stand)
        i = 0
        while i < np.shape(text_rect)[0]:
            # 向下合并, 注意越界问题
            if i < np.shape(text_rect)[0] - 1 and text_rect[i][2] == text_rect[i + 1][2]:
                if float(text_rect[i][1] - text_rect[i][0]) < h_stand and text_rect[i + 1][0] - text_rect[i][1]\
                        < 0.6 * dis_stand:
                    text_rect[i][1] = text_rect[i + 1][1]
                    text_rect = np.delete(text_rect, i + 1, axis=0)
                    continue
                # 考虑字形较高问题， 要求更近的间距判断
                elif float(text_rect[i][1] - text_rect[i][0]) < 0.8 * height and text_rect[i + 1][0] - text_rect[i][
                    1] < 0.5 * dis_stand:
                    text_rect[i][1] = text_rect[i + 1][1]
                    text_rect = np.delete(text_rect, i + 1, axis=0)
                    continue
            # 向上合并
            if float(text_rect[i][1] - text_rect[i][0]) < h_stand and text_rect[i - 1][2] == text_rect[i][2] and \
                    text_rect[i][0] - text_rect[i - 1][1] < 0.5 * dis_stand:
                text_rect[i - 1][1] = text_rect[i][1]
                text_rect = np.delete(text_rect, i, axis=0)
                continue
            i = i + 1

        for i in range(np.shape(text_rect)[0]):
            print("每字高度：", text_rect[i][1] - text_rect[i][0])

        # 在原图上绘制截图矩形区域
        rectangle_img = cv2.rectangle(src, (text_rect[0][2], text_rect[0][0]), (text_rect[0][3], text_rect[0][1]),
                                      (0, 0, 255), thickness=2)
        for i, rect_roi in enumerate(text_rect):
            rectangle_img = cv2.rectangle(src, (rect_roi[2], rect_roi[0]), (rect_roi[3], rect_roi[1]), (255, 0, 0),
                                          thickness=2)
            cv2.putText(rectangle_img, str(i + 1), (rect_roi[3] - 5, rect_roi[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 255), 2)
        self.__outImageRGB = rectangle_img
        self.__drawImage(self.srcImageView, self.__outImageRGB)

    def __projection2(self, img):
        src = img[:, :, [2, 1, 0]]
        src = src.copy()
        img = src
        # 输出图像尺寸和通道信息
        sp = img.shape
        print("图像信息：", sp)
        sz1 = sp[0]  # height(rows) of image
        sz2 = sp[1]  # width(columns) of image
        sz3 = sp[2]  # the pixels value is made up of three primary colors
        print('width: %d \n height: %d \n number: %d' % (sz2, sz1, sz3))
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        retval, threshold_img = cv2.threshold(gray_img, 100, 255, cv2.THRESH_BINARY_INV)

        # 垂直投影分割图像
        gray_value_y = []
        for i in range(sz2):
            white_value = 0
            for j in range(sz1):
                if threshold_img[j, i] == 255:
                    white_value += 1
            gray_value_y.append(white_value)
        print("", gray_value_y)
        # 创建图像显示垂直投影分割图像结果
        veri_projection_img = np.zeros((sp[0], sp[1], 1), np.uint8)
        for i in range(sz2):
            for j in range(gray_value_y[i]):
                veri_projection_img[j, i] = 255
        # cv2.imwrite(os.path.join(save_path, "veri_projection_img.jpg"), veri_projection_img)
        text_rect = []

        # 根据垂直投影分割识别列
        inline_y = 0
        start_y = 0
        text_rect_y = []
        for i in range(len(gray_value_y)):
            if inline_y == 0 and gray_value_y[i] > 1:
                inline_y = 1
                start_y = i
            elif inline_y == 1 and gray_value_y[i] < 2 and (i - start_y) > 0:
                inline_y = 0
                if i - start_y > 0:
                    rect = [start_y, i + 1]
                    text_rect_y.append(rect)
            elif inline_y == 1 and i == len(gray_value_y) - 1:
                inline_y = 0
                if i - start_y > 1:
                    rect = [start_y - 1, i + 1]
                    text_rect_y.append(rect)
        text_rect_y.reverse()
        print("分列区域，每列数据起始位置Y：", text_rect_y)
        # 每列数据分段
        # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 3))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilate_img = cv2.dilate(threshold_img, kernel)
        # cv2.imwrite(os.path.join(save_path, "dilate_img.jpg"), dilate_img)
        for rect in text_rect_y:
            cropImg = dilate_img[0:sp[0], rect[0]:rect[1]]  # 裁剪图像y-start:y-end,x-start:x-end
            sp_x = cropImg.shape
            # 水平投影分割图像
            gray_value_x = []
            for i in range(sp_x[0]):
                white_value = 0
                for j in range(sp_x[1]):
                    if cropImg[i, j] == 255:
                        white_value += 1
                gray_value_x.append(white_value)
            # 创建图像显示水平投影分割图像结果
            hori_projection_img = np.zeros((sp_x[0], sp_x[1], 1), np.uint8)
            for i in range(sp_x[0]):
                for j in range(gray_value_x[i]):
                    veri_projection_img[i, j] = 255
            # cv2.imwrite(os.path.join(save_path, "hori.jpg"), hori_projection_img)
            # 根据水平投影分割识别行
            inline_x = 0
            start_x = 0
            text_rect_x = []
            ind = 0
            for i in range(len(gray_value_x)):
                ind += 1
                if inline_x == 0 and gray_value_x[i] > 10:
                    inline_x = 1
                    start_x = i
                elif inline_x == 1 and gray_value_x[i] < 2 and (i - start_x) > 10:
                    inline_x = 0
                    if i - start_x > 10:
                        rect_x = [start_x - 1, i + 1]
                        text_rect_x.append(rect_x)
                        text_rect.append([start_x - 1, i + 1, rect[0], rect[1]])
                        # cropImg_rect = threshold_img[start_x - 1:i + 1, rect[0]:rect[1]]  # 裁剪二值化图像
                        # crop_img = img[start_x - 1:i + 1, rect[0]:rect[1]] #裁剪原图像
                        # cv2.imshow("cropImg_rect", cropImg_rect)
                        # cv2.imwrite(os.path.join(crop_path,str(ind)+".jpg"),crop_img)
                        # break
                # break

        sum = 0
        dis = 0
        cnt = 0

        # 获得平均高度
        for i in range(np.shape(text_rect)[0]):
            sum += text_rect[i][1] - text_rect[i][0]
        height = sum / np.shape(text_rect)[0]
        print("height:", height)

        # 获取平均间距
        for i in range(np.shape(text_rect)[0] - 1):
            if text_rect[i][2] == text_rect[i + 1][2]:
                dis += text_rect[i + 1][0] - text_rect[i][1]
                cnt += 1
                i += 1
            else:
                i += 1
        dis = dis / cnt
        print("distance:", dis)

        h_stand = 0.7 * height  # 小于认为存在分离问题
        dis_stand = dis  # 判断向上或向下合并（默认向下）
        print("h_stand:", h_stand)
        print("dis_stand:", dis_stand)
        i = 0
        while i < np.shape(text_rect)[0]:
            # 向下合并, 注意越界问题
            if i < np.shape(text_rect)[0] - 1 and text_rect[i][2] == text_rect[i + 1][2]:
                if float(text_rect[i][1] - text_rect[i][0]) < h_stand and text_rect[i + 1][0] - text_rect[i][1] < \
                        dis_stand:
                    text_rect[i][1] = text_rect[i + 1][1]
                    text_rect = np.delete(text_rect, i + 1, axis=0)
                    continue
                # 考虑字形较高问题， 要求更近的间距判断
                elif float(text_rect[i][1] - text_rect[i][0]) < 0.8 * height and text_rect[i + 1][0] - text_rect[i][
                    1] < 0.5 * dis_stand:
                    text_rect[i][1] = text_rect[i + 1][1]
                    text_rect = np.delete(text_rect, i + 1, axis=0)
                    continue
            # 向上合并
            if float(text_rect[i][1] - text_rect[i][0]) < h_stand and text_rect[i - 1][2] == text_rect[i][2] and \
                    text_rect[i][0] - text_rect[i - 1][1] < dis_stand:
                text_rect[i - 1][1] = text_rect[i][1]
                text_rect = np.delete(text_rect, i, axis=0)
                continue
            i = i + 1

        for i in range(np.shape(text_rect)[0]):
            print("每字高度：", text_rect[i][1] - text_rect[i][0])

        # 在原图上绘制截图矩形区域
        rectangle_img = cv2.rectangle(src, (text_rect[0][2], text_rect[0][0]), (text_rect[0][3], text_rect[0][1]),
                                      (0, 0, 255), thickness=2)
        for i, rect_roi in enumerate(text_rect):
            rectangle_img = cv2.rectangle(src, (rect_roi[2], rect_roi[0]), (rect_roi[3], rect_roi[1]), (255, 0, 0),
                                          thickness=2)
            cv2.putText(rectangle_img, str(i + 1), (rect_roi[3] - 5, rect_roi[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (100, 0, 255), 2)
        self.__interImageRGB = rectangle_img
        self.__drawImage(self.interpretView, self.__interImageRGB)

    # 弹窗事件
    def popWindow3(self):
        self.form2 = QWidget()
        self.ui2 = Ui_Form()
        self.ui2.setupUi(self.form2)
        name = os.path.splitext(self.__Name)[0]
        path = "D:/infor/" + name + "-3.txt"
        file = open(path, encoding="utf_8")
        infor = file.read()
        self.ui2.textEdit.append(infor)
        file.close()
        self.form2.show()

    def popWindow1(self):
        self.form3 = QWidget()
        self.ui3 = Ui_Form1()
        self.ui3.setupUi(self.form3)
        name = os.path.splitext(self.__Name)[0]
        path = "D:/infor/" + name + "-1.txt"
        file = open(path, encoding="utf_8")
        infor = file.read()
        self.ui3.textEdit.append(infor)
        file.close()
        self.form3.show()


    def popWindow2(self):
        self.form4 = QWidget()
        self.ui4 = Ui_Form2()
        self.ui4.setupUi(self.form4)
        name = os.path.splitext(self.__Name)[0]
        path = "D:/infor/" + name + "-2.txt"
        file = open(path, encoding="utf_8")
        infor = file.read()
        self.ui4.textEdit.append(infor)
        file.close()
        self.form4.show()

    def popsrc(self):
        self.form5 = QWidget()
        self.ui5 = Ui_srcForm()
        self.ui5.setupUi(self.form5)
        self.form5.show()
        srcName = "D:/src/" + self.__Name
        __srcImg = cv2.imread(srcName)
        self.__srcRGB = cv2.cvtColor(__srcImg, cv2.COLOR_BGR2RGB)
        self.__drawImage(self.ui5.src, self.__srcRGB)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    # 实例化主窗口
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec())
