# import torch.nn as nn
# import torch
# from torch.autograd import Variable
# import torchvision.models as models
# import torch.nn.functional as F
# from utils.metric import seg_metric
# from dataset import *
# import os
# import cv2


# epoch_size = 200
# MODEL_PATH = "model/params.pkl"

# class DACBlock(nn.Module):
# 	"""docstring for DACBlock"""
# 	def __init__(self, channel):
# 		super(DACBlock, self).__init__()
# 		self.conv_1 = nn.Conv2d(channel, channel, 3, dilation=1, padding=1)
# 		self.conv_3 = nn.Conv2d(channel, channel, 3, dilation=3, padding=3)
# 		self.conv_5 = nn.Conv2d(channel, channel, 3, dilation=5, padding=5)
# 		self.conv_1_1 = nn.Conv2d(channel, channel, 1, bias=False)

# 	def forward(self, x):
# 		out1 = F.relu(self.conv_1(x), inplace=True)
# 		out2 = F.relu(self.conv_1_1(self.conv_3(x)), inplace=True)
# 		out3 = F.relu(self.conv_1_1(self.conv_3(self.conv_1(x))), inplace=True)
# 		out4 = F.relu(self.conv_1_1(self.conv_5(self.conv_3(self.conv_1(x)))), inplace=True)
# 		out = x + out1 + out2 + out3 + out4
# 		return out

# class RMPBlock(nn.Module):
# 	"""docstring for RMPBlock"""
# 	def __init__(self, channel):
# 		super(RMPBlock, self).__init__()

# 		# 1×1 convolution
# 		self.conv_1_1 = self.conv_1_1 = nn.Conv2d(channel, 1, 1)

# 		self.upsample = nn.Upsample(size=(16,16), mode='bilinear',align_corners=False)
# 		# self.upsample3 = nn.ConvTranspose2d(1,1,kernel_size=16,stride=8,padding=4)


# 	def forward(self, x):
# 		pool1 = F.max_pool2d(x, 2)
# 		up1 = self.upsample(self.conv_1_1(pool1))
# 		pool2 = F.max_pool2d(x, 3)
# 		up2 = self.upsample(self.conv_1_1(pool2))
# 		pool3 = F.max_pool2d(x, 5)
# 		up3 = self.upsample(self.conv_1_1(pool3))
# 		pool4 = F.max_pool2d(x, 6)
# 		up4 = self.upsample(self.conv_1_1(pool4))

# 		out = torch.cat([up1, up2, up3, up4, x], dim=1)

# 		return out

# class DecodeBlock(nn.Module):
# 	"""docstring for DecodeBlock"""
# 	def __init__(self, in_ch, out_ch):
# 		super(DecodeBlock, self).__init__()
# 		self.conv1_1_1 = self.conv_1_1 = nn.Conv2d(in_ch, out_ch, 1)
# 		self.upsample = nn.ConvTranspose2d(out_ch,out_ch,4,2,1)
# 		self.conv2_1_1 = self.conv_1_1 = nn.Conv2d(out_ch, out_ch, 1)

# 	def forward(self, x):
# 		out = self.conv1_1_1(x)
# 		out = self.upsample(out)
# 		out = self.conv2_1_1(out)

# 		return out
		

# class CENet(nn.Module):
# 	"""docstring for CENet"""
# 	def __init__(self, in_ch, out_ch):
# 		super(CENet, self).__init__()
# 		resnet = models.resnet18(pretrained=True)
# 		new_model = nn.Sequential(*list(resnet.children())[0:-2])

# 		self.resblock1 = new_model[0:3]
# 		self.resblock2 = new_model[3:5]
# 		self.resblock3 = new_model[5]
# 		self.resblock4 = new_model[6]
# 		self.resblock5 = new_model[7]

# 		self.dac = DACBlock(512)
# 		self.rmp = RMPBlock(512)

# 		self.decode1 = DecodeBlock(516,256)
# 		self.decode2 = DecodeBlock(512,128)
# 		self.decode3 = DecodeBlock(256,64) 
# 		self.decode4 = DecodeBlock(128,64)
# 		self.decode5 = DecodeBlock(128,64)

# 		self.conv = nn.Conv2d(64,out_ch, 1)


# 	def forward(self, x):

# 		# feature encoder
# 		down1 = self.resblock1(x)
# 		down2 = self.resblock2(down1)
# 		down3 = self.resblock3(down2)
# 		down4 = self.resblock4(down3)
# 		down5 = self.resblock5(down4)

# 		# context extractor
# 		rmp = self.rmp(self.dac(down5))

# 		# feature decoder 
# 		up1 = torch.cat([self.decode1(rmp),down4], dim=1)
# 		up2 = torch.cat([self.decode2(up1),down3], dim=1)
# 		up3 = torch.cat([self.decode3(up2),down2], dim=1)
# 		up4 = torch.cat([self.decode4(up3),down1], dim=1)
# 		up5 = self.decode5(up4)

# 		out = self.conv(up5)
# 		prob = torch.sigmoid(out)

# 		return prob

# 	def predict(self, img):
# 		x = x_transform(img).unsqueeze(0)
# 		x = Variable(x.float().cuda())
# 		output = self.forward(x).squeeze().cpu().data.numpy()
# 		result_img = np.zeros(output.shape, dtype=np.uint8)
# 		result_img[output>0.5]=255

# 		return result_img


# class DiscLoss(nn.Module):
# 	"""docstring for DiscLoss"""
# 	def __init__(self):
# 		super(DiscLoss, self).__init__()
		
# 	def forward(self, output, target):
# 		iou1 = torch.sum(output*target)/ \
# 						(torch.sum(output)+torch.sum(target)+1e-6)
# 		iou2 = torch.sum((1-output)*(1-target))/ \
# 				(torch.sum((1-output))+torch.sum((1-target))+1e-6)

# 		return 1-iou1-iou2

# if __name__ == '__main__':
# 	model = CENet(3,1).cuda()

# 	# load the pretrained model
# 	if os.path.exists(MODEL_PATH):
# 		model.load_state_dict(torch.load(MODEL_PATH))

# 	criterion = nn.BCELoss()
# 	# criterion = DiscLoss()
# 	optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)

# 	for i in range(epoch_size):
# 		epoch_loss = 0.0; epoch_count = 0   # epoch_loss表示每轮的损失总和

# 		for img, label in data_loader:
# 			epoch_count += 1 
# 			x = Variable(img.float().cuda())
# 			y = Variable(label.float().cuda())

# 			output = model(x)
# 			loss = criterion(output, y)

# 			optimizer.zero_grad()
# 			loss.backward()
# 			optimizer.step()
# 			epoch_loss += loss 

# 		avg_loss = (epoch_loss.cpu().data.numpy())/epoch_count

 
# 		if i%2==0:
# 			torch.save(model.state_dict(), MODEL_PATH)
# 			img = Image.open("data/test/image/40.png")
# 			gt = cv2.imread("data/test/label_black/40.png", cv2.IMREAD_GRAYSCALE)
# 			result = model.predict(img)
# 			# precision, recall, acc = seg_metric(result, gt)
# 			print("[INFO]epoch = %d, loss = %g"%(i, avg_loss)) 
# 			# print("[metric]precision = {:.2%}, recall = {:.2%}, acc = {:.2%},"\
#    #          	.format(precision,recall,acc)) 
# 			cv2.imwrite("result.png", result)



import torch.nn as nn
import torch
from torch.autograd import Variable
import torchvision.models as models
import torch.nn.functional as F
from utils.metric import seg_metric
from dataset import *
import os
import cv2

from functools import partial

import Constants

nonlinearity = partial(F.relu, inplace=True)

epoch_size = 300
MODEL_PATH = "model/cenetparams.pkl"

class DACblock(nn.Module):
    def __init__(self, channel):
        super(DACblock, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=3, padding=3)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=5, padding=5)
        self.conv1x1 = nn.Conv2d(channel, channel, kernel_size=1, dilation=1, padding=0)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.conv1x1(self.dilate2(x)))
        dilate3_out = nonlinearity(self.conv1x1(self.dilate2(self.dilate1(x))))
        dilate4_out = nonlinearity(self.conv1x1(self.dilate3(self.dilate2(self.dilate1(x)))))
        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out
        return out


class DACblock_without_atrous(nn.Module):
    def __init__(self, channel):
        super(DACblock_without_atrous, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.conv1x1 = nn.Conv2d(channel, channel, kernel_size=1, dilation=1, padding=0)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.conv1x1(self.dilate2(x)))
        dilate3_out = nonlinearity(self.conv1x1(self.dilate2(self.dilate1(x))))
        dilate4_out = nonlinearity(self.conv1x1(self.dilate3(self.dilate2(self.dilate1(x)))))
        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out

        return out

class DACblock_with_inception(nn.Module):
    def __init__(self, channel):
        super(DACblock_with_inception, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=1, dilation=1, padding=0)

        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.conv1x1 = nn.Conv2d(2 * channel, channel, kernel_size=1, dilation=1, padding=0)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.dilate3(self.dilate1(x)))
        dilate_concat = nonlinearity(self.conv1x1(torch.cat([dilate1_out, dilate2_out], 1)))
        dilate3_out = nonlinearity(self.dilate1(dilate_concat))
        out = x + dilate3_out
        return out


class DACblock_with_inception_blocks(nn.Module):
    def __init__(self, channel):
        super(DACblock_with_inception_blocks, self).__init__()
        self.conv1x1 = nn.Conv2d(channel, channel, kernel_size=1, dilation=1, padding=0)
        self.conv3x3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.conv5x5 = nn.Conv2d(channel, channel, kernel_size=5, dilation=1, padding=2)
        self.pooling = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        dilate1_out = nonlinearity(self.conv1x1(x))
        dilate2_out = nonlinearity(self.conv3x3(self.conv1x1(x)))
        dilate3_out = nonlinearity(self.conv5x5(self.conv1x1(x)))
        dilate4_out = self.pooling(x)
        out = dilate1_out + dilate2_out + dilate3_out + dilate4_out
        return out



class PSPModule(nn.Module):
    def __init__(self, features, out_features=1024, sizes=(2, 3, 6, 14)):
        super().__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, size) for size in sizes])
        self.bottleneck = nn.Conv2d(features * (len(sizes) + 1), out_features, kernel_size=1)
        self.relu = nn.ReLU()

    def _make_stage(self, features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, features, kernel_size=1, bias=False)
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.upsample(input=stage(feats), size=(h, w), mode='bilinear') for stage in self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return self.relu(bottle)


class SPPblock(nn.Module):
    def __init__(self, in_channels):
        super(SPPblock, self).__init__()
        self.pool1 = nn.MaxPool2d(kernel_size=[2, 2], stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=[3, 3], stride=3)
        self.pool3 = nn.MaxPool2d(kernel_size=[5, 5], stride=5)
        self.pool4 = nn.MaxPool2d(kernel_size=[6, 6], stride=6)

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=1, kernel_size=1, padding=0)

    def forward(self, x):
        self.in_channels, h, w = x.size(1), x.size(2), x.size(3)
        self.layer1 = F.upsample(self.conv(self.pool1(x)), size=(h, w), mode='bilinear',align_corners=False)
        self.layer2 = F.upsample(self.conv(self.pool2(x)), size=(h, w), mode='bilinear',align_corners=False)
        self.layer3 = F.upsample(self.conv(self.pool3(x)), size=(h, w), mode='bilinear',align_corners=False)
        self.layer4 = F.upsample(self.conv(self.pool4(x)), size=(h, w), mode='bilinear',align_corners=False)

        out = torch.cat([self.layer1, self.layer2, self.layer3, self.layer4, x], 1)

        return out


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(DecoderBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nonlinearity

        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nonlinearity

        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nonlinearity

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x


class CENet(nn.Module):
    def __init__(self, num_classes=Constants.BINARY_CLASS, num_channels=3):
        super(CENet, self).__init__()

        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=True)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.dblock = DACblock(512)
        self.spp = SPPblock(512)

        self.decoder4 = DecoderBlock(516, filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

    def forward(self, x):
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # Center
        e4 = self.dblock(e4)
        e4 = self.spp(e4)

        # Decoder
        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)

        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return torch.sigmoid(out)


    def predict(self, img):
    	x = x_transform(img).unsqueeze(0)
    	x = Variable(x.float().cuda())
    	output = self.forward(x).squeeze().cpu().data.numpy()
    	result_img = np.zeros(output.shape, dtype=np.uint8)
    	result_img[output>0.5]=255

    	return result_img

class DiscLoss(nn.Module):
    def __init__(self, batch=True):
        super(DiscLoss, self).__init__()
        self.batch = batch

    def soft_dice_coeff(self, y_pred, y_true):
        smooth = 0.0  # may change
        if self.batch:
            i = torch.sum(y_true)
            j = torch.sum(y_pred)
            intersection = torch.sum(y_true * y_pred)
        else:
            i = y_true.sum(1).sum(1).sum(1)
            j = y_pred.sum(1).sum(1).sum(1)
            intersection = (y_true * y_pred).sum(1).sum(1).sum(1)
        score = (2. * intersection + smooth) / (i + j + smooth)
        # score = (intersection + smooth) / (i + j - intersection + smooth)#iou
        return score.mean()

    def forward(self, y_true, y_pred):
        loss = 1 - self.soft_dice_coeff(y_true, y_pred)
        return loss

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()

        resnet = models.resnet18(pretrained=True)

    def forward(self, x):
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        return e4

if __name__ == '__main__':
    model = models.resnet18(pretrained=True)
    new_model = nn.Sequential(*list(model.children())[0:-2])
    # print(new_model)
    x = torch.rand(1,3,512,512)
    for i, m in enumerate(new_model.children()):
        x = m(x)
        # print(i, x.shape)

    model = CENet().cuda()

    # load the pretrained model
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH))

    criterion = nn.BCELoss()
    # criterion = DiscLoss()
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
        #     img = change_image_channels(Image.open("data4/test/image/40.png"))
        #     gt = cv2.imread("data4/test/label_black/40.png", cv2.IMREAD_GRAYSCALE)
        #     result = model.predict(img)
        #     precision, recall, acc, mIOU,F1 = seg_metric(result, gt)
        #     print("[INFO]epoch = %d, loss = %g" % (i, avg_loss))
        #     print("[metric]precision = {:.2%}, recall = {:.2%}, acc = {:.2%},mIOU= {:.2%},F1= {:.2%}" \
        #           .format(precision, recall, acc, mIOU, F1))
        #     cv2.imwrite("result.png", result)
        # if i % 2 == 0:
        #     img = change_image_channels(Image.open("data4/test/image/48.png"))
        #     gt = cv2.imread("data4/test/label_black/48.png", cv2.IMREAD_GRAYSCALE)
        #     result = model.predict(img)
        #     precision, recall, acc, mIOU, F1,Dice,Jaccard = seg_metric(result, gt)
        #     print("[INFO]epoch = %d, loss = %g" % (i, avg_loss))
        #     print("[metric]precision = {:.2%}, recall = {:.2%}, acc = {:.2%}, mIOU= {:.2%}, F1= {:.2%},Dice= {:.2%},Jaccard= {:.2%}" \
        #         .format(precision, recall, acc, mIOU, F1, Dice, Jaccard))
        # if F1 > 0.8:
        #     torch.save(model.state_dict(), MODEL_PATH)
        #     break\
        img = change_image_channels(Image.open("data4/test/image/49.png"))
        gt = cv2.imread("data4/test/label_black/49.png", cv2.IMREAD_GRAYSCALE)
        result = model.predict(img)
        precision, recall, acc, mIOU, F1 = seg_metric(result, gt)
        print("[INFO]epoch = %d, loss = %g" % (i, avg_loss))
        print("[metric]precision = {:.2%}, recall = {:.2%}, acc = {:.2%},mIOU= {:.2%},F1= {:.2%}" \
              .format(precision, recall, acc, mIOU, F1))
        if F1 > 0.6:
            torch.save(model.state_dict(), MODEL_PATH)
            break