import numpy as np
from matplotlib import transforms
from torch.utils.data import Dataset
import cv2
import  xml_text

class VOC2012(Dataset):
    def __init__(self,is_train=True,is_aug=True):
        """
        :param is_train: 调用的是训练集(True)，还是验证集(False)
        :param is_aug:  是否进行数据增广
        """
        self.filenames = []  # 储存数据集的文件名称
        if is_train:
            with open("E:\dataset\VOCdevkit\VOC2012/" + "ImageSets/Main/train.txt", 'r') as f: # 调用包含训练集图像名称的txt文件
                self.filenames = [x.strip() for x in f]
        else:
            with open("E:\dataset\VOCdevkit\VOC2012/" + "ImageSets/Main/val.txt", 'r') as f:
                self.filenames = [x.strip() for x in f]
        self.imgpath = "E:\dataset\VOCdevkit\VOC2012/" + "JPEGImages/"  # 原始图像所在的路径
        self.labelpath = "./labels/"  # 图像对应的label文件(.txt文件)的路径
        self.is_aug = is_aug

    def __len__(self):
        return len(self.filenames)#返回的是训练数据集的大小

    def __getitem__(self, item):
        img = cv2.imread(self.imgpath+self.filenames[item]+".jpg")  # 读取原始图像   item代表的是list索引值
        h,w = img.shape[0:2]
        input_size = 448  # 输入YOLOv1网络的图像尺寸为448x448
        # 因为数据集内原始图像的尺寸是不定的，所以需要进行适当的padding，将原始图像padding成宽高一致的正方形
        # 然后再将Padding后的正方形图像缩放成448x448
        padw, padh = 0, 0  # 要记录宽高方向的padding具体数值，因为padding之后需要调整bbox的位置信息
        if h>w:
            padw = (h - w) // 2
            img = np.pad(img,((0,0),(padw,padw),(0,0)),'constant',constant_values=0)   # ‘constant’——表示连续填充相同的值，每个轴可以分别指定填充值，constant_values=（x, y）时前面用x填充，后面用y填充，缺省值填充0
        elif w>h:
            padh = (w - h) // 2
            img = np.pad(img,((padh,padh),(0,0),(0,0)), 'constant', constant_values=0)
        img = cv2.resize(img,(input_size,input_size))
        # 图像增广部分，这里不做过多处理，因为改变bbox信息还蛮麻烦的

        if self.is_aug:
            aug = transforms.Compose([
                transforms.ToTensor()
            ])
            img = aug(img)

        # 读取图像对应的bbox信息，按1维的方式储存，每5个元素表示一个bbox的(cls,xc,yc,w,h)
        with open(self.labelpath+self.filenames[item]+".txt") as f:
            bbox = f.read().split('\n')#换行符
        bbox = [x.split() for x in bbox]#空格符
        bbox = [float(x) for y in bbox for x in y]   #二维数组
        if len(bbox)%5!=0:
            raise ValueError("File:"+self.labelpath+self.filenames[item]+".txt"+"——bbox Extraction Error!")

        # 根据padding、图像增广等操作，将原始的bbox数据转换为修改后图像的bbox数据    原始的坐标根据增广的方向对坐标进行改变
        for i in range(len(bbox)//5):
            if padw != 0:
                bbox[i * 5 + 1] = (bbox[i * 5 + 1] * w + padw) / h
                bbox[i * 5 + 3] = (bbox[i * 5 + 3] * w) / h
            elif padh != 0:
                bbox[i * 5 + 2] = (bbox[i * 5 + 2] * h + padh) / w
                bbox[i * 5 + 4] = (bbox[i * 5 + 4] * h) / w
            # 此处可以写代码验证一下，查看padding后修改的bbox数值是否正确，在原图中画出bbox检验

        labels = convert_bbox2labels(bbox)  # 将所有bbox的(cls,x,y,w,h)数据转换为训练时方便计算Loss的数据形式(7,7,5*B+cls_num)
        # 此处可以写代码验证一下，经过convert_bbox2labels函数后得到的labels变量中储存的数据是否正确
        labels = transforms.ToTensor()(labels)
        return img,labels


def convert_bbox2labels(bbox):
    """将bbox的(cls,x,y,w,h)数据转换为训练时方便计算Loss的数据形式(7,7,5*B+cls_num)
    注意，输入的bbox的信息是(xc,yc,w,h)格式的，转换为labels后，bbox的信息转换为了(px,py,w,h)格式"""
    gridsize = 1.0/7  #yolo会将格子分为7*7的样式 为一个格子的宽度 格子为正方形的
    labels = np.zeros((7,7,5*2+len(xml_text.CLASSES)))  # 注意，此处需要根据不同数据集的类别个数进行修改
    for i in range(len(bbox)//5):  #每个循环都是一个标注选框
        gridx = int(bbox[i*5+1] // gridsize)  # 当前bbox中心落在第gridx个网格,列
        gridy = int(bbox[i*5+2] // gridsize)  # 当前bbox中心落在第gridy个网格,行
        # (bbox中心坐标 - 网格左上角点的坐标)/网格大小  ==> bbox中心点的相对位置
        gridpx = bbox[i * 5 + 1] / gridsize - gridx  #一个格子的大小为1/7   大小为这个
        gridpy = bbox[i * 5 + 2] / gridsize - gridy
        # 将第gridy行，gridx列的网格设置为负责当前ground truth的预测，置信度和对应类别概率均置为1
        labels[gridy, gridx, 0:5] = np.array([gridpx, gridpy, bbox[i * 5 + 3], bbox[i * 5 + 4], 1])
        labels[gridy, gridx, 5:10] = np.array([gridpx, gridpy, bbox[i * 5 + 3], bbox[i * 5 + 4], 1])
        labels[gridy, gridx, 10+int(bbox[i*5])] = 1
    return labels
def  test2():
    train_data = VOC2012()
    print(train_data)
test2()