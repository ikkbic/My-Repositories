import os
import cv2
import csv

import torch
from PIL import Image
from utils.metric import seg_metric
from dataset import change_image_channels
from dataset import change_image_channels_noclahe
# from res_unet import Unet
# from InceptionRes import Unet
# from InceptionRes1 import Unet
# from InceptionRes2 import Unet
# from attention_unet import Unet
from ResAttentionU import Unet
# from IncepResAttenRes import Unet
# from IncepResAttenRes_paint import Unet
# from unet1 import Unet
# from unet import Unet
# from cenet import CENet
# from AttenRes import Unet
# from Res import Unet
import numpy as np



# import matplotlib.pyplot as plt
# from sklearn import svm
# from sklearn.metrics import roc_curve, auc  ###计算roc和auc
# from sklearn import cross_validation
# from scipy.sparse.csgraph import connected_components


# MODEL_PATH = "model/unet1_params.pkl"
MODEL_PATH = "model/attention_res_params_d30.pkl"
IMG_PATH = "data3/test/image"
GT_PATH = "data3/test/label_black"





if __name__ == '__main__':
	model = Unet().cuda()
	# model = CENet().cuda()
	body_dic = []
	if os.path.exists(MODEL_PATH):    # 预加载模型参数
		model.load_state_dict(torch.load(MODEL_PATH))

	img_names = os.listdir(IMG_PATH)
	label_names = os.listdir(GT_PATH)
	i = 1
	for image_name, gt_name in zip(img_names, label_names):
		image_path = os.path.join(IMG_PATH, image_name)
		gt_path = os.path.join(GT_PATH, gt_name)
		# img = change_image_channels_noclahe(Image.open(image_path))
		img = change_image_channels(Image.open(image_path))
		gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
		result = model.predict(img)

		precision, recall, acc,Dice,Jaccard = seg_metric(result, gt)

		# header = ['pre', 'recall', 'acc', 'mIOU', 'F1']

		da = {'pre': precision, 'recall': recall, 'acc': acc, 'Dice':Dice,'Jaccard':Jaccard}
		# print(da)

		body_dic.append(da)
		# print(body_dic)



		print("[metric]precision = {:.2%}, recall = {:.2%}, acc = {:.2%}, Dice= {:.2%},Jaccard= {:.2%}" \
			  .format(precision, recall, acc, Dice,Jaccard))
		cv2.imwrite("result/%s.png" % image_name.split('.')[0], result)
		print("Finish %d images." % (i ))
		i += 1
		cv2.imshow("result", result)
		cv2.imshow("gt", gt)
		image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
		cv2.imshow("image", image)
		if cv2.waitKey(0) == 27: break
		# cv2.imwrite("0.jpg", image)


	header = ['pre', 'recall', 'acc', 'Dice','Jaccard']
	with open(file='AttenRes_d300.csv', mode='w', encoding='utf-8', newline='') as f:
		write = csv.DictWriter(f, fieldnames=header)
		write.writeheader()
		write.writerows(body_dic)










