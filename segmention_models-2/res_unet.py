import torch.nn as nn
import torch
from torch.autograd import Variable
import torchvision.models as models
from dataset import *
import os
import cv2
from PIL import Image
from utils.metric import seg_metric

epoch_size = 400
MODEL_PATH = "model/res_params.pkl"

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
    def __init__(self):
        super(Unet, self).__init__()
        resnet18 = models.resnet18()
        resnet = nn.Sequential(*list(resnet18.children())[0:-2])
        self.encode1 = resnet[0:3]
        self.encode2 = resnet[3:5]
        self.encode3 = resnet[5]
        self.encode4 = resnet[6]

        self.conv5 = DoubleConv(256, 512)
        self.up6 = nn.ConvTranspose2d(512, 128, 2, stride=2)
        self.conv6 = DoubleConv(256, 128)
        self.up7 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv7 = DoubleConv(128, 128)
        self.up8 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv8 = DoubleConv(128, 64)
        self.up9 = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.conv9 = nn.Conv2d(64, 1, 1)

    def forward(self,x):
        c1=self.encode1(x)
        c2=self.encode2(c1)
        c3=self.encode3(c2)
        c4=self.encode4(c3)
        c5 = self.conv5(c4)
        up_6= self.up6(c5)
        merge6 = torch.cat([up_6, c3], dim=1)
        c6=self.conv6(merge6)
        up_7=self.up7(c6)
        merge7 = torch.cat([up_7, c2], dim=1)
        c7=self.conv7(merge7)
        up_8=self.up8(c7)
        merge8 = torch.cat([up_8, c1], dim=1)
        c8=self.conv8(merge8)
        up_9=self.up9(c8)
        c9=self.conv9(up_9)
        out = nn.Sigmoid()(c9)
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
    model = Unet().cuda()
    ans = 0
    if os.path.exists(MODEL_PATH):    # 预加载模型参数
        model.load_state_dict(torch.load(MODEL_PATH))
    criterion = nn.BCELoss()
    # criterion = SoftDiceLoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
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
        #     img = change_image_channels(Image.open("data4/test/image/40.png"))
        #     gt = cv2.imread("data4/test/label_black/40.png", cv2.IMREAD_GRAYSCALE)
        #     result = model.predict(img)
        #     precision, recall, acc, mIOU, F1 = seg_metric(result, gt)
        #     print("[INFO]epoch = %d, loss = %g" % (i, avg_loss))
        #     print("[metric]precision = {:.2%}, recall = {:.2%}, acc = {:.2%},mIOU= {:.2%},F1= {:.2%}" \
        #           .format(precision, recall, acc, mIOU, F1))
        #     cv2.imwrite("result.png", result)
        if i % 2 == 0:
            img = change_image_channels(Image.open("data4/test/image/49.png"))
            gt = cv2.imread("data4/test/label_black/49.png", cv2.IMREAD_GRAYSCALE)
            result = model.predict(img)
            precision, recall, acc, mIOU, F1, Dice, Jaccard= seg_metric(result, gt)
            print("[INFO]epoch = %d, loss = %g" % (i, avg_loss))
            print(
                "[metric]precision = {:.2%}, recall = {:.2%}, acc = {:.2%}, mIOU= {:.2%}, F1= {:.2%},Dice= {:.2%},Jaccard= {:.2%}" \
                .format(precision, recall, acc, mIOU, F1, Dice, Jaccard))
        if F1 > 0.7:
            torch.save(model.state_dict(), MODEL_PATH)
            break
        # print("[INFO]epoch = %d, loss = %g" % (i, avg_loss))
        #
        # img = change_image_channels(Image.open("data4/test/image/49.png"))
        # gt = cv2.imread("data4/test/label_black/49.png", cv2.IMREAD_GRAYSCALE)
        # result = model.predict(img)
        # precision, recall, acc, mIOU, F1,Dice,Jaccard = seg_metric(result, gt)
        # print(
        #     "[metric]precision = {:.2%}, recall = {:.2%}, acc = {:.2%}, mIOU= {:.2%}, F1= {:.2%},Dice= {:.2%},Jaccard= {:.2%}" \
        #     .format(precision, recall, acc, mIOU, F1, Dice, Jaccard))
        #
        # img1 = change_image_channels(Image.open("data4/test/image/48.png"))
        # gt1 = cv2.imread("data4/test/label_black/48.png", cv2.IMREAD_GRAYSCALE)
        # result1 = model.predict(img1)
        # precision1, recall1, acc1, mIOU1, F1_1,Dice,Jaccard = seg_metric(result1, gt1)
        #
        #
        # img2 = change_image_channels(Image.open("data4/test/image/47.png"))
        # gt2 = cv2.imread("data4/test/label_black/47.png", cv2.IMREAD_GRAYSCALE)
        # result2 = model.predict(img2)
        # precision2, recall2, acc2, mIOU2, F1_2,Dice,Jaccard = seg_metric(result2, gt2)
        #
        #
        # img3 = change_image_channels(Image.open("data4/test/image/46.png"))
        # gt3 = cv2.imread("data4/test/label_black/46.png", cv2.IMREAD_GRAYSCALE)
        # result3 = model.predict(img3)
        # precision3, recall3, acc3, mIOU3, F1_3,Dice,Jaccard = seg_metric(result3, gt3)
        #
        # img4 = change_image_channels(Image.open("data4/test/image/45.png"))
        # gt4 = cv2.imread("data4/test/label_black/45.png", cv2.IMREAD_GRAYSCALE)
        # result4 = model.predict(img4)
        # precision4, recall4, acc4, mIOU4, F1_4,Dice,Jaccard = seg_metric(result4, gt4)
        #
        #
        # img5 = change_image_channels(Image.open("data4/test/image/44.png"))
        # gt5 = cv2.imread("data4/test/label_black/44.png", cv2.IMREAD_GRAYSCALE)
        # result5 = model.predict(img5)
        # precision5, recall5, acc5, mIOU5, F1_5,Dice,Jaccard = seg_metric(result5, gt5)
        #
        #
        # img6 = change_image_channels(Image.open("data4/test/image/43.png"))
        # gt6 = cv2.imread("data4/test/label_black/43.png", cv2.IMREAD_GRAYSCALE)
        # result6 = model.predict(img6)
        # precision6, recall6, acc6, mIOU6, F1_6,Dice,Jaccard = seg_metric(result6, gt6)
        #
        # img7 = change_image_channels(Image.open("data4/test/image/42.png"))
        # gt7 = cv2.imread("data4/test/label_black/42.png", cv2.IMREAD_GRAYSCALE)
        # result7 = model.predict(img7)
        # precision7, recall7, acc7, mIOU7, F1_7,Dice,Jaccard = seg_metric(result7, gt7)
        #
        # img8 = change_image_channels(Image.open("data4/test/image/41.png"))
        # gt8 = cv2.imread("data4/test/label_black/41.png", cv2.IMREAD_GRAYSCALE)
        # result8 = model.predict(img8)
        # precision8, recall8, acc8, mIOU8, F1_8,Dice,Jaccard = seg_metric(result8, gt8)
        #
        # img9 = change_image_channels(Image.open("data4/test/image/40.png"))
        # gt9 = cv2.imread("data4/test/label_black/40.png", cv2.IMREAD_GRAYSCALE)
        # result9 = model.predict(img9)
        # precision9, recall9, acc9, mIOU9, F1_9,Dice,Jaccard = seg_metric(result9, gt9)
        #
        #
        # sumF1 = F1 + F1_1 + F1_2 + F1_3 + F1_4 + F1_5 + F1_6 + F1_7 + F1_8 + F1_9
        # if ans == 0:
        #     ans = sumF1
        # if sumF1 > ans :
        #     ans = sumF1
        #     torch.save(model.state_dict(), MODEL_PATH)