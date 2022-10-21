import torch.nn as nn
import torch
from torch.autograd import Variable
from dataset import *
import os
import cv2
from PIL import Image
from utils.metric import seg_metric
import csv


epoch_size = 100
MODEL_PATH = "model/attention_res_params_d1-paint.pkl"


class BasicBlock(nn.Module):
    def __init__(self, channel):
        super(BasicBlock, self).__init__()
        self.conv3_3 = nn.Conv2d(channel, channel, 3, padding=1, bias=False)
        self.conv1_1 = nn.Conv2d(channel, channel, 1, bias=False)

    def forward(self, x):
        return self.conv3_3(x) + self.conv1_1(x)

class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi

class ResConnection(nn.Module):
    def __init__(self, channel, num_layer):
        super(ResConnection, self).__init__()
        self.channel = channel
        self.num_layer = num_layer
        self.model = self._make_layer()

    def _make_layer(self):
        layers = []
        for i in range(self.num_layer):
            layers.append(BasicBlock(self.channel))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

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
    def __init__(self, in_ch=3, out_ch=1):
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
        self.att1 = Attention_block(512, 512, 256)
        self.conn1 = ResConnection(512, 1)
        self.conv6 = DoubleConv(512, 512)
        self.up7 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.att2 = Attention_block(256, 256, 128)
        self.conn2 = ResConnection(256, 2)
        self.conv7 = DoubleConv(256, 256)
        self.up8 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.att3 = Attention_block(128, 128, 64)
        self.conn3 = ResConnection(128, 3)
        self.conv8 = DoubleConv(128, 128)
        self.up9 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.att4 = Attention_block(64, 64, 32)
        self.conn4 = ResConnection(64, 4)
        self.conv9 = DoubleConv(64, 64)
        self.conv10 = nn.Conv2d(64, out_ch, 1)

    def forward(self, x):
        c1 = self.conv1(x)
        p1 = self.pool1(c1)
        c2 = self.conv2(p1)
        p2 = self.pool2(c2)
        c3 = self.conv3(p2)
        p3 = self.pool3(c3)
        c4 = self.conv4(p3)
        p4 = self.pool4(c4)
        c5 = self.conv5(p4)
        up_6 = self.up6(c5)
        x1 = self.att1(up_6, self.conn1(c4))
        merge6 = torch.add(up_6, x1)
        c6 = self.conv6(merge6)
        up_7 = self.up7(c6)
        x2 = self.att2(up_7, self.conn2(c3))
        merge7 = torch.add(up_7, x2)
        c7 = self.conv7(merge7)
        up_8 = self.up8(c7)
        x3 = self.att3(up_8, self.conn3(c2))
        merge8 = torch.add(up_8, x3)
        c8 = self.conv8(merge8)
        up_9 = self.up9(c8)
        x4 = self.att4(up_9, self.conn4(c1))
        merge9 = torch.add(up_9, x4)
        c9 = self.conv9(merge9)
        c10 = self.conv10(c9)
        out = nn.Sigmoid()(c10)
        return out

    def predict(self, img):
        x = x_transform(img).unsqueeze(0)
        x = Variable(x.float().cuda())
        output = self.forward(x).squeeze().cpu().data.numpy()
        result_img = np.zeros(output.shape, dtype=np.uint8)
        result_img[output > 0.5] = 255
        return result_img


class MyLoss(nn.Module):
    """docstring for MyLoss"""
    def __init__(self):
        super(MyLoss, self).__init__()

    def forward(self, output, target):
        intersection = torch.sum(output*target)
        iou = torch.sum(torch.pow(output,2)) + torch.sum(torch.pow(target,2)) - intersection

        loss = 1 - intersection/iou
        # print(intersection)
        # print(iou)
        return loss


class CustomLoss(torch.nn.Module):

    def __init__(self):
        super(CustomLoss, self).__init__()

    def forward(self,inputs, targets, beta=0.9, weights=None):

        batch_size = targets.size(0)
        loss = 0.0

        for i in range(batch_size):
            prob = inputs[i]
            ref = targets[i]

            alpha = 1.0 - beta

            tp = (ref * prob).sum()
            fp = ((1 - ref) * prob).sum()
            fn = (ref * (1 - prob)).sum()
            tversky = tp / (tp + alpha * fp + beta * fn)
            loss = loss + (1 - tversky)
        return loss / batch_size



if __name__ == '__main__':
    model = Unet().cuda()
    ans = 0.6
    LOSS_dic = []
    DICEMEAN_dic = []
    header1 = ['loss']
    header2 = ['Dice_mean']
    Dice_all = 0
    IMG_PATH = "data1/test/image"
    GT_PATH = "data1/test/label_black"
    img_names = os.listdir(IMG_PATH)
    label_names = os.listdir(GT_PATH)
    if os.path.exists(MODEL_PATH):    # 预加载模型参数
        model.load_state_dict(torch.load(MODEL_PATH))
    criterion = nn.BCELoss()
    # criterion = SoftDiceLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    for i in range(epoch_size):
        epoch_loss = 0.0; epoch_count = 0   # epoch_loss表示每轮的损失总和
        for img, label in data_loader:
            epoch_count += 1
            x = Variable(img.float().cuda())
            y = Variable(label.float().cuda())

            output = model(x)
            loss = criterion(output, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss
        avg_loss = (epoch_loss.cpu().data.numpy())/epoch_count
        if i % 1 == 0:
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
    with open(file='AttenRes_d1_loss.csv', mode='w', encoding='utf-8', newline='') as f:
        write = csv.DictWriter(f, fieldnames=header1)
        write.writeheader()
        write.writerows(LOSS_dic)
    with open(file='AttenRes_d1_dicemean.csv', mode='w', encoding='utf-8', newline='') as f:
        write = csv.DictWriter(f, fieldnames=header2)
        write.writeheader()
        write.writerows(DICEMEAN_dic)
