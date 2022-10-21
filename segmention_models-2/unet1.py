import torch.nn as nn
import torch
from torch.autograd import Variable
from dataset import *
import os
import cv2
from PIL import Image
from utils.metric import seg_metric


epoch_size =200
MODEL_PATH = "model/unet1_params_d0.pkl"

class BasicBlock(nn.Module):
    def __init__(self, channel):
        super(BasicBlock, self).__init__()
        self.conv3_3 = nn.Conv2d(channel,channel,3,padding=1, bias=False)
        self.conv1_1 = nn.Conv2d(channel,channel,1, bias=False)

    def forward(self, x):
        return self.conv3_3(x) + self.conv1_1(x)

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
        self.conn1 = ResConnection(512, 1)
        self.conv6 = DoubleConv(512, 512)
        self.up7 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conn2 = ResConnection(256, 2)
        self.conv7 = DoubleConv(256, 256)
        self.up8 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conn3 = ResConnection(128, 3)
        self.conv8 = DoubleConv(128, 128)
        self.up9 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conn4 = ResConnection(64, 4)
        self.conv9 = DoubleConv(64, 64)
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
        merge6 = torch.add(up_6, self.conn1(c4))
        c6=self.conv6(merge6)
        up_7=self.up7(c6)
        merge7 = torch.add(up_7, self.conn2(c3))
        c7=self.conv7(merge7)
        up_8=self.up8(c7)
        merge8 = torch.add(up_8, self.conn3(c2))
        c8=self.conv8(merge8)
        up_9=self.up9(c8)
        merge9=torch.add(up_9,self.conn4(c1))
        c9=self.conv9(merge9)
        c10=self.conv10(c9)
        out = nn.Sigmoid()(c10)
        return out

    def predict(self, img):
        x = x_transform(img).unsqueeze(0)
        x = Variable(x.float().cuda())
        output = self.forward(x).squeeze().cpu().data.numpy()
        result_img = np.zeros(output.shape, dtype=np.uint8)
        result_img[output>0.5]=255
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
    ans1 = ans2 = 0.6
    model = Unet().cuda()
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
        # if i%2==0:
        #     torch.save(model.state_dict(), MODEL_PATH)
        #     img = change_image_channels(Image.open("data4/test/image/49.png"))
        #     gt = cv2.imread("data4/test/label_black/49.png", cv2.IMREAD_GRAYSCALE)
        #     result = model.predict(img)
        #     precision, recall, acc, mIOU, F1 = seg_metric(result, gt)
        #     print("[INFO]epoch = %d, loss = %g" % (i, avg_loss))
        #     print("[metric]precision = {:.2%}, recall = {:.2%}, acc = {:.2%},mIOU= {:.2%},F1= {:.2%}" \
        #           .format(precision, recall, acc, mIOU, F1))
        #     cv2.imwrite("result.png", result)

        # if i % 2 == 0:
        if i % 2 == 0:
            print("[INFO]epoch = %d, loss = %g" % (i, avg_loss))
            da = {'loss': avg_loss, }
            img_first = change_image_channels(Image.open("data0/test/image/0.png"))
            gt_first = cv2.imread("data0/test/label_black/0.png", cv2.IMREAD_GRAYSCALE)
            result = model.predict(img_first)
            precision_first, recall_first, acc_first,Dice_first, Jaccard_first = seg_metric(result, gt_first)
            print("[metric]precision = {:.2%}, recall = {:.2%}, acc = {:.2%}, Dice= {:.2%},Jaccard= {:.2%}" \
                .format(precision_first, recall_first, acc_first,  Dice_first, Jaccard_first))
            img_second = change_image_channels(Image.open("data0/test/image/1.png"))
            gt_second = cv2.imread("data0/test/label_black/1.png", cv2.IMREAD_GRAYSCALE)
            result_second = model.predict(img_second)
            precision_second, recall_second, acc_second, Dice_second, Jaccard_second = seg_metric(result_second, gt_second)
            print("[metric]precision = {:.2%}, recall = {:.2%}, acc = {:.2%}, Dice= {:.2%},Jaccard= {:.2%}" \
                .format(precision_second, recall_second, acc_second, Dice_second, Jaccard_second))
            if Dice_first +Dice_second > ans2 + ans1:
                ans1 = Dice_first
                ans2 = Dice_second
                torch.save(model.state_dict(), MODEL_PATH)
            # break

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
        # img2 = change_image_channels(Image.open("data4/test/image/47.png"))
        # gt2 = cv2.imread("data4/test/label_black/47.png", cv2.IMREAD_GRAYSCALE)
        # result2 = model.predict(img2)
        # precision2, recall2, acc2, mIOU2, F1_2 = seg_metric(result2, gt2)
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
        # img5 = change_image_channels(Image.open("data4/test/image/44.png"))
        # gt5 = cv2.imread("data4/test/label_black/44.png", cv2.IMREAD_GRAYSCALE)
        # result5 = model.predict(img5)
        # precision5, recall5, acc5, mIOU5, F1_5 = seg_metric(result5, gt5)
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
        # sumF1 = F1 + F1_1 + F1_2 + F1_3 + F1_4 + F1_5 + F1_6 + F1_7 + F1_8 + F1_9
        # if ans == 0:
        #     ans = sumF1
        # if sumF1 > ans and F1 > 0.75 and F1_3 > 0.75:
        #     ans = sumF1
        #     torch.save(model.state_dict(), MODEL_PATH)


