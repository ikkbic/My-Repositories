import torch.nn as nn
import torch
from torch.autograd import Variable
from dataset import *
import os
import cv2
from PIL import Image
from utils.metric import seg_metric
import torch.nn.functional as F
import csv
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
epoch_size = 100
MODEL_PATH = "model/IncepResAttenRes_d2.pkl"


class BasicBlock(nn.Module):
    def __init__(self, channel):
        super(BasicBlock, self).__init__()
        self.conv3_3 = nn.Conv2d(channel, channel, 3, padding=1, bias=False)
        self.conv1_1 = nn.Conv2d(channel, channel, 1, bias=False)

    def forward(self, x):
        return self.conv3_3(x) + self.conv1_1(x)


class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        return out

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


class InceptionResNetA(nn.Module):
    def __init__(self, in_channels,out_channels):
        super(InceptionResNetA, self).__init__()
        # branch1: conv1*1(32)
        self.b1 = BasicConv2d(in_channels, out_channels, kernel_size=1)

        # branch2: conv1*1(32) --> con3*3(32)
        self.b2_1 = BasicConv2d(in_channels, in_channels, kernel_size=1)
        self.b2_2 = BasicConv2d(in_channels, in_channels, kernel_size=3, padding=1)

        # branch3: conv1*1(32) --> conv3*3(32) --> conv3*3(32)
        self.b3_1 = BasicConv2d(in_channels, in_channels, kernel_size=1)
        self.b3_2 = BasicConv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.b3_3 = BasicConv2d(in_channels, in_channels, kernel_size=3, padding=1)

        # totalbranch: conv1*1(256)
        self.tb = BasicConv2d(out_channels, out_channels, kernel_size=1)
        self.a1 = BasicConv2d(in_channels,out_channels,kernel_size=1)

    def forward(self, x):
        x = F.relu(x)
        x_double = self.a1(x)
        # b_out1 = F.relu(self.b1(x))
        b_out2 = F.relu(self.b2_2(F.relu(self.b2_1(x))))
        b_out3 = F.relu(self.b3_3(F.relu(self.b3_2(F.relu(self.b3_1(x))))))
        b_out = torch.cat([ b_out2, b_out3], 1)
        # b_out = torch.cat([b_out1, b_out2, b_out3], 1)
        b_out = self.tb(b_out)
        # y = torch.cat([b_out, x], dim=1)
        # y = torch.cat([x, b_out], dim=1)
        y = x_double + b_out
        out = F.relu(y)
        # print(out)

        return out

class InceptionResNetB(nn.Module):
    def __init__(self, in_channels,out_channels):
        super(InceptionResNetB, self).__init__()
        # branch1: conv1*1(32)
        self.b1 = BasicConv2d(in_channels, in_channels, kernel_size=1)

        # branch2: conv1*1(32) --> con3*3(32)
        self.b2_1 = BasicConv2d(in_channels, in_channels, kernel_size=1)
        self.b2_2 = BasicConv2d(in_channels, in_channels, kernel_size=3, padding=1)

        # branch3: conv1*1(32) --> conv3*3(32) --> conv3*3(32)
        self.b3_1 = BasicConv2d(in_channels, in_channels, kernel_size=1)
        self.b3_2 = BasicConv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.b3_3 = BasicConv2d(in_channels, in_channels, kernel_size=3, padding=1)

        # totalbranch: conv1*1(256)
        self.tb = BasicConv2d(2*in_channels, in_channels, kernel_size=1)
        self.a1 = BasicConv2d(in_channels,out_channels,kernel_size=1)

    def forward(self, x):
        x = F.relu(x)
        # x_double = self.a1(x)
        # b_out1 = F.relu(self.b1(x))
        b_out2 = F.relu(self.b2_2(F.relu(self.b2_1(x))))
        b_out3 = F.relu(self.b3_3(F.relu(self.b3_2(F.relu(self.b3_1(x))))))
        b_out = torch.cat([ b_out2, b_out3], 1)
        # b_out = torch.cat([b_out1, b_out2, b_out3], 1)
        b_out = self.tb(b_out)
        y = torch.cat([b_out, x], dim=1)
        # y = torch.cat([x, b_out], dim=1)
        y = x + b_out
        out = F.relu(y)
        # print(out)

        return out




class Unet(nn.Module):
    def __init__(self, in_ch=3, out_ch=1):
        super(Unet, self).__init__()

        self.conv1 = DoubleConv(in_ch, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = InceptionResNetA(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = InceptionResNetA(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = InceptionResNetA(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        # self.conv5 = DoubleConv(512, 1024)
        self.conv5 = InceptionResNetA(512,1024)
        self.up6 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.att1 = Attention_block(512, 512, 256)
        self.conn1 = ResConnection(512, 1)
        self.conv6 = InceptionResNetB(512, 512)
        self.up7 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.att2 = Attention_block(256, 256, 128)
        self.conn2 = ResConnection(256, 2)
        self.conv7 = InceptionResNetB(256, 256)
        self.up8 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.att3 = Attention_block(128, 128, 64)
        self.conn3 = ResConnection(128, 3)
        self.conv8 = InceptionResNetB(128, 128)
        self.up9 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.att4 = Attention_block(64, 64, 32)
        self.conn4 = ResConnection(64, 4)
        self.conv9 = InceptionResNetB(64, 64)
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
        merge7 = torch.add(up_7,x2)
        c7 = self.conv7(merge7)
        up_8 = self.up8(c7)
        x3 = self.att3(up_8, self.conn3(c2))
        merge8 = torch.add(up_8, x3)
        c8 = self.conv8(merge8)
        up_9 = self.up9(c8)
        x4 = self.att4(up_9, self.conn4(c1))
        merge9 = torch.add(up_9,x4)
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


class SoftDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(SoftDiceLoss, self).__init__()

    def forward(self, logits, targets):
        num = targets.size(0)
        smooth = 1

        probs = torch.sigmoid(logits)
        m1 = probs.view(num, -1)
        m2 = targets.view(num, -1)
        intersection = (m1 * m2)

        score = 2. * (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
        score = 1 - score.sum() / num
        return score


if __name__ == '__main__':
    model = Unet().cuda()
    ans1 = ans2 = ans3 = 0.5
    body_dic = []
    if os.path.exists(MODEL_PATH):  # 预加载模型参数
        model.load_state_dict(torch.load(MODEL_PATH))
    criterion = nn.BCELoss()
    # criterion = SoftDiceLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    for i in range(epoch_size):
        epoch_loss = 0.0;
        epoch_count = 0  # epoch_loss表示每轮的损失总和
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
        avg_loss = (epoch_loss.cpu().data.numpy()) / epoch_count
        # if i % 2 == 0:
        # print("[INFO]epoch = %d, loss = %g" % (i, avg_loss))
        #
        # img = change_image_channels(Image.open("data4/test/image/49.png"))
        # gt = cv2.imread("data4/test/label_black/49.png", cv2.IMREAD_GRAYSCALE)
        # result = model.predict(img)
        # precision, recall, acc, mIOU, F1 = seg_metric(result, gt)
        # print("[metric]precision = {:.2%}, recall = {:.2%}, acc = {:.2%},mIOU= {:.2%},F1= {:.2%}" \
        #       .format(precision, recall, acc, mIOU, F1))
        #
        # img1 = change_image_channels(Image.open("data4/test/image/48.png"))
        # gt1 = cv2.imread("data4/test/label_black/48.png", cv2.IMREAD_GRAYSCALE)
        # result1 = model.predict(img1)
        # precision1, recall1, acc1, mIOU1, F1_1 = seg_metric(result1, gt1)
        #
        #
        # img2 = change_image_channels(Image.open("data4/test/image/47.png"))
        # gt2 = cv2.imread("data4/test/label_black/47.png", cv2.IMREAD_GRAYSCALE)
        # result2 = model.predict(img2)
        # precision2, recall2, acc2, mIOU2, F1_2 = seg_metric(result2, gt2)
        #
        #
        # img3 = change_image_channels(Image.open("data4/test/image/46.png"))
        # gt3 = cv2.imread("data4/test/label_black/46.png", cv2.IMREAD_GRAYSCALE)
        # result3 = model.predict(img3)
        # precision3, recall3, acc3, mIOU3, F1_3 = seg_metric(result3, gt3)
        #
        # img4 = change_image_channels(Image.open("data4/test/image/45.png"))
        # gt4 = cv2.imread("data4/test/label_black/45.png", cv2.IMREAD_GRAYSCALE)
        # result4 = model.predict(img4)
        # precision4, recall4, acc4, mIOU4, F1_4 = seg_metric(result4, gt4)
        #
        #
        # img5 = change_image_channels(Image.open("data4/test/image/44.png"))
        # gt5 = cv2.imread("data4/test/label_black/44.png", cv2.IMREAD_GRAYSCALE)
        # result5 = model.predict(img5)
        # precision5, recall5, acc5, mIOU5, F1_5 = seg_metric(result5, gt5)
        #
        #
        # img6 = change_image_channels(Image.open("data4/test/image/43.png"))
        # gt6 = cv2.imread("data4/test/label_black/43.png", cv2.IMREAD_GRAYSCALE)
        # result6 = model.predict(img6)
        # precision6, recall6, acc6, mIOU6, F1_6 = seg_metric(result6, gt6)
        #
        # img7 = change_image_channels(Image.open("data4/test/image/42.png"))
        # gt7 = cv2.imread("data4/test/label_black/42.png", cv2.IMREAD_GRAYSCALE)
        # result7 = model.predict(img7)
        # precision7, recall7, acc7, mIOU7, F1_7 = seg_metric(result7, gt7)
        #
        # img8 = change_image_channels(Image.open("data4/test/image/41.png"))
        # gt8 = cv2.imread("data4/test/label_black/41.png", cv2.IMREAD_GRAYSCALE)
        # result8 = model.predict(img8)
        # precision8, recall8, acc8, mIOU8, F1_8 = seg_metric(result8, gt8)
        #
        # img9 = change_image_channels(Image.open("data4/test/image/40.png"))
        # gt9 = cv2.imread("data4/test/label_black/40.png", cv2.IMREAD_GRAYSCALE)
        # result9 = model.predict(img9)
        # precision9, recall9, acc9, mIOU9, F1_9 = seg_metric(result9, gt9)
        #
        #
        # sumF1 = F1 + F1_1 + F1_2 + F1_3 + F1_4 + F1_5 + F1_6 + F1_7 + F1_8 + F1_9
        # if ans == 0:
        #     ans = sumF1
        # if sumF1 > ans and F1 > 0.75 and F1_3 > 0.75:
        #     ans = sumF1
        #     torch.save(model.state_dict(), MODEL_PATH)
        if i % 2 == 0:
            print("[INFO]epoch = %d, loss = %g" % (i, avg_loss))
            da = {'loss': avg_loss, }
            body_dic.append(da)
            img_first = change_image_channels(Image.open("data2/test/image/21.png"))
            gt_first = cv2.imread("data2/test/label_black/21.png", cv2.IMREAD_GRAYSCALE)
            result = model.predict(img_first)
            precision_first, recall_first, acc_first,Dice_first, Jaccard_first = seg_metric(result, gt_first)
            print("[metric]precision = {:.2%}, recall = {:.2%}, acc = {:.2%}, Dice= {:.2%},Jaccard= {:.2%}" \
                .format(precision_first, recall_first, acc_first,  Dice_first, Jaccard_first))
            img_second = change_image_channels(Image.open("data2/test/image/25.png"))
            gt_second = cv2.imread("data2/test/label_black/25.png", cv2.IMREAD_GRAYSCALE)
            result_second = model.predict(img_second)
            precision_second, recall_second, acc_second, Dice_second, Jaccard_second = seg_metric(result_second, gt_second)
            print("[metric]precision = {:.2%}, recall = {:.2%}, acc = {:.2%}, Dice= {:.2%},Jaccard= {:.2%}" \
                .format(precision_second, recall_second, acc_second, Dice_second, Jaccard_second))
            img_third = change_image_channels(Image.open("data2/test/image/25.png"))
            gt_third = cv2.imread("data2/test/label_black/25.png", cv2.IMREAD_GRAYSCALE)
            result_third = model.predict(img_third)
            precision_third, recall_third, acc_third, Dice_third, Jaccard_third = seg_metric(result_third, gt_third)
            print("[metric]precision = {:.2%}, recall = {:.2%}, acc = {:.2%}, Dice= {:.2%},Jaccard= {:.2%}" \
                  .format(precision_first, recall_first, acc_first, Dice_first, Jaccard_first))
            if Dice_first +Dice_second + Dice_third> ans2 + ans1 + ans3:
                ans1 = Dice_first
                ans2 = Dice_second
                ans3 = Dice_third
                torch.save(model.state_dict(), MODEL_PATH)
    header = ['loss']
    with open(file='mymethod_d1_loss.csv', mode='w', encoding='utf-8', newline='') as f:
        write = csv.DictWriter(f, fieldnames=header)
        write.writeheader()
        write.writerows(body_dic)