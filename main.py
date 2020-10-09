# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import torch
import visdom
from torch.utils.data import DataLoader

from dataset import VOC2012
from loss import Loss_yolov1
from resnet import YOLOv1_resnet
import numpy as np

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    epoch = 50
    batchsize = 5
    lr = 0.01

    train_data = VOC2012()
    train_dataloader = DataLoader(VOC2012(is_train=True),batch_size=batchsize,shuffle=True)

    model = YOLOv1_resnet().cuda()
    # model.children()里是按模块(Sequential)提取的子模块，而不是具体到每个层，具体可以参见pytorch帮助文档
    # 冻结resnet34特征提取层，特征提取层不参与参数更新
    for layer in model.children():
        layer.requires_grad = False
        break
    criterion = Loss_yolov1()
    optimizer = torch.optim.SGD(model.parameters(),lr=lr,momentum=0.9,weight_decay=0.0005)

    is_vis = False  # 是否进行可视化，如果没有visdom可以将其设置为false
    if is_vis:
        vis = visdom.Visdom()
        viswin1 = vis.line(np.array([0.]),np.array([0.]),opts=dict(title="Loss/Step",xlabel="100*step",ylabel="Loss"))

    for e in range(epoch):
        model.train()
        yl = torch.Tensor([0]).cuda()
        for i,(inputs,labels) in enumerate(train_dataloader):
            inputs = inputs.cuda()
            labels = labels.float().cuda()
            pred = model(inputs)
            loss = criterion(pred, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print("Epoch %d/%d| Step %d/%d| Loss: %.2f"%(e,epoch,i,len(train_data)//batchsize,loss))
            yl = yl + loss
            if is_vis and (i+1)%100==0:
                vis.line(np.array([yl.cpu().item()/(i+1)]),np.array([i+e*len(train_data)//batchsize]),win=viswin1,update='append')
        if (e+1)%10==0:
            torch.save(model,"./models_pkl/YOLOv1_epoch"+str(e+1)+".pkl")
            # compute_val_map(model)


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
