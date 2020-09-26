import xml.etree.ElementTree as ET
import os
import cv2
import matplotlib.pyplot as plt
CLASSES = ['person', 'bird', 'cat', 'cow', 'dog', 'horse', 'sheep',
           'aeroplane', 'bicycle', 'boat', 'bus', 'car', 'motorbike', 'train',
           'bottle', 'chair', 'dining table', 'potted plant', 'sofa', 'tvmonitor']

def convert(size, box):
    """将bbox的左上角点、右下角点坐标的格式，转换为bbox中心点+bbox的w,h的格式
    并进行归一化"""
    dw = 1. / size[0]
    dh = 1. / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


def convert_annotation(image_id):
    """把图像image_id的xml文件转换为目标检测的label文件(txt)
    其中包含物体的类别，bbox的左上角点坐标以及bbox的宽、高
    并将四个物理量归一化"""
    in_file = open("E:\dataset\VOCdevkit\VOC2012/" + 'Annotations/%s' % (image_id))
    image_id = image_id.split('.')[0]#返回的内容是 2012_004010  图片的id
    out_file = open('./labels/%s.txt' % (image_id), 'w')
    tree = ET.parse(in_file)   #导入数据
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)  #图像的高宽
    h = int(size.find('height').text)

    for obj in root.iter('object'):   #集合数据类型如list、dict、str等是Iterable但不是Iterator，不过可以通过iter()函数获得一个Iterator对象。
        difficult = obj.find('difficult').text
        cls = obj.find('name').text  #图片名字
        if cls not in CLASSES or int(difficult) == 1:
            continue
        cls_id = CLASSES.index(cls)  #获取对应的下标
        xmlbox = obj.find('bndbox')
        points = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
             float(xmlbox.find('ymax').text))
        bb = convert((w, h), points)  #获取坐标归一化的结果
        out_file.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')    #cls——ID 对应标签力数组的id


def make_label_txt():
    """在labels文件夹下创建image_id.txt，对应每个image_id.xml提取出的bbox信息"""
    filenames = os.listdir("E:\dataset\VOCdevkit\VOC2012/" + 'Annotations')
    for file in filenames:   #file的格式是“2012_004010.xml”
      convert_annotation(file)

def show_labels_img(imgname):
    """imgname是输入图像的名称，无下标"""
    img = cv2.imread("E:\dataset\VOCdevkit\VOC2012/" + "JPEGImages/" + imgname + ".jpg")
    h, w = img.shape[:2]
    print(w,h)
    label = []
    with open("./labels/"+imgname+".txt",'r') as flabel:
        for label in flabel:
            label = label.split(' ')
            label = [float(x.strip()) for x in label]  #strip 移除首尾空格
            print(CLASSES[int(label[0])])
            pt1 = (int(label[1] * w - label[3] * w / 2), int(label[2] * h - label[4] * h / 2))# 左上角的坐标
            pt2 = (int(label[1] * w + label[3] * w / 2), int(label[2] * h + label[4] * h / 2))
            cv2.putText(img,CLASSES[int(label[0])],pt1,cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255))
            """
            puttext(src,str,Point2d(pt.at<double>(i,0),pt.at<double>(i,1)),CV_FONT_HERSHEY_SIMPLEX,3,Scalar(0,0,255),6,8);        //pt是mat类型，n行2列的矩阵
            第一个参数是：需要写字的原图像，第二个：需要写的内容，string类型的；
            第三个：需要写的内容的左下角的坐标 第五个：字体大小 第六个：颜色第七个：字体的厚度 第八个：默认8
            """
            cv2.rectangle(img,pt1,pt2,(0,0,255,2))

    cv2.imshow("img",img)
    cv2.waitKey(0)

def test():
    img = cv2.imread("E:\dataset\VOCdevkit\VOC2012/" + "JPEGImages/" + "2007_000175" + ".jpg")
    cv2.imshow("img",img)
    cv2.waitKey(0)
    print(img.shape)
    fruits = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    fruits_all = [fruits, fruits[:, :, 0], fruits[:, :, 1], fruits[:, :, 2]]
    channels = ["RGB", "red", "green", "blue"]
    for i in range(4):
        plt.subplot(2, 2, i + 1)
        plt.imshow(fruits_all[i], cmap=plt.cm.gray)
        plt.title(channels[i])
    plt.show()
    for fruit in fruits_all:
        print(fruit.shape)

def test2():
    with open("E:\workplace\pycharm\yolov1\labels/2007_000033.txt") as f:
        bbox = f.read().split('\n')  # 换行符
    bbox = [x.split() for x in bbox]  # 空格符
    bbox = [float(x) for y in bbox for x in y]  # 二维数组
    print(bbox)
test2()