import torch.nn as nn
import torch
from torch.autograd import Variable
from dataset import *
import os
import cv2
from PIL import Image
from utils.metric import seg_metric
import csv

epoch_size = 500
MODEL_PATH = "model/unet_paramsmycl_d0.pkl"


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)


class Unet(nn.Module):
    def __init__(self,in_ch=3,out_ch=1):
        super(Unet, self).__init__()

        self.conv1 = DoubleConv(in_ch, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        self.conv5 = DoubleConv(512, 1024)
        self.up6 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.conv6 = DoubleConv(1024, 512)
        self.up7 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv7 = DoubleConv(512, 256)
        self.up8 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv8 = DoubleConv(256, 128)
        self.up9 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv9 = DoubleConv(128, 64)
        self.conv10 = nn.Conv2d(64,out_ch, 1)

    def forward(self,x):
        c1=self.conv1(x)
        p1=self.pool1(c1)
        c2=self.conv2(p1)
        p2=self.pool2(c2)
        c3=self.conv3(p2)
        p3=self.pool3(c3)
        c4=self.conv4(p3)
        p4=self.pool4(c4)
        c5=self.conv5(p4)
        up_6= self.up6(c5)
        merge6 = torch.cat([up_6, c4], dim=1)
        c6=self.conv6(merge6)
        up_7=self.up7(c6)
        merge7 = torch.cat([up_7, c3], dim=1)
        c7=self.conv7(merge7)
        up_8=self.up8(c7)
        merge8 = torch.cat([up_8, c2], dim=1)
        c8=self.conv8(merge8)
        up_9=self.up9(c8)
        merge9=torch.cat([up_9,c1],dim=1)
        c9=self.conv9(merge9)
        c10=self.conv10(c9)
        out = nn.Sigmoid()(c10)
        return out

    def predict(self, img):
        x = x_transform(img).unsqueeze(0)
        x = Variable(x.float().cuda())
        output = self.forward(x).squeeze().cpu().data.numpy()
        result_img = np.zeros(output.shape, dtype=np.uint8)
        result_img[output > 0.5] = 255
        return result_img

if __name__ == '__main__':
    ans = 0.6
    LOSS_dic = []
    DICEMEAN_dic = []
    header1 = ['loss']
    header2 = ['Dice_mean']
    Dice_all = 0
    IMG_PATH = "data2/test/image"
    GT_PATH = "data2/test/label_black"
    img_names = os.listdir(IMG_PATH)
    label_names = os.listdir(GT_PATH)
    model = Unet(3,1).cuda()
    if os.path.exists(MODEL_PATH):  # 预加载模型参数
        model.load_state_dict(torch.load(MODEL_PATH))
    criterion = nn.BCELoss()
    # criterion = MyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    for i in range(epoch_size):
        epoch_loss = 0.0;
        epoch_count = 0  # epoch_loss表示每轮的损失总和
        for img, label in data_loader:
            epoch_count += 1
            x = Variable(img.float().cuda())
            # print(x.shape)
            y = Variable(label.float().cuda())

            output = model(x)
            loss = criterion(output, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss
        avg_loss = (epoch_loss.cpu().data.numpy()) / epoch_count

        if i % 2 == 0:
            print("[INFO]epoch = %d, loss = %g" % (i, avg_loss))
            losscsv = {'loss': avg_loss, }
            LOSS_dic.append(losscsv)
            for image_name, gt_name in zip(img_names, label_names):
                image_path = os.path.join(IMG_PATH, image_name)
                gt_path = os.path.join(GT_PATH, gt_name)
                img = change_image_channels(Image.open(image_path))
                gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
                result = model.predict(img)

                precision, recall, acc, Dice, Jaccard = seg_metric(result, gt)
                Dice_all = Dice_all + Dice

                da = {'pre': precision, 'recall': recall, 'acc': acc, 'Dice': Dice, 'Jaccard': Jaccard}

                print("[metric]precision = {:.2%}, recall = {:.2%}, acc = {:.2%}, Dice= {:.2%},Jaccard= {:.2%}" \
                      .format(precision, recall, acc, Dice, Jaccard))
                cv2.imwrite("result/%s.png" % image_name.split('.')[0], result)
                i += 1
                image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            Dice_mean = Dice_all / 10
            Dice_all = 0
            print("Dice_mean= {:.2%}".format(Dice_mean))
            Dicecsv = {'Dice_mean': Dice_mean}
            DICEMEAN_dic.append(Dicecsv)

            if Dice_mean > ans:
                ans = Dice_mean
                torch.save(model.state_dict(), MODEL_PATH)
    with open(file='unet_d2_loss.csv', mode='w', encoding='utf-8', newline='') as f:
        write = csv.DictWriter(f, fieldnames=header1)
        write.writeheader()
        write.writerows(LOSS_dic)
    with open(file='unet_d2_dicemean.csv', mode='w', encoding='utf-8', newline='') as f:
        write = csv.DictWriter(f, fieldnames=header2)
        write.writeheader()
        write.writerows(DICEMEAN_dic)