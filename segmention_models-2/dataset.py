# from torchvision.transforms import transforms
# from torch.utils.data import Dataset, DataLoader
# import PIL.Image as Image
# import numpy as np
# import os
# import cv2
# import random
# from PIL import Image
#
# ROOT_PATH = "data4/train"
# input_size = [2112,112]
# # input_size = [448, 448]
#
# def change_image_channels(image):
#     if image.mode == 'RGBA':
#         r, g, b, a = image.split()
#         image = Image.merge("RGB", (r, g, b))
#     img = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
#     # cv2.imshow("img1", img)
#     b, g, r = cv2.split(img)
#     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#     b = clahe.apply(b)
#     g = clahe.apply(g)
#     r = clahe.apply(r)
#     img = cv2.merge([b, g, r])
#     # cv2.imshow("img2", img)
#     # cv2.waitKey()
#     return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#
#
# def load_image_path(root_path):
#     img_pairs = []
#     raw_list = os.listdir(root_path + "/image")
#     label_list = os.listdir(root_path + "/label_black")
#     for raw_name, label_name in zip(raw_list, label_list):
#         img_path = os.path.join(root_path,"image",raw_name)
#         label_path = os.path.join(root_path,"label_black",label_name)
#         img_pairs.append([img_path, label_path])
#     return img_pairs
#
#
# def random_crop(img, label, crop_size):
# 	size = img.size
# 	x_start = random.randint(0, size[0]-crop_size[0])
# 	y_start = random.randint(0, size[1]-crop_size[1])
# 	img = img.crop([x_start, y_start, x_start+crop_size[0], y_start+crop_size[1]])
# 	label = label.crop([x_start, y_start, x_start+crop_size[0], y_start+crop_size[1]])
# 	return img, label
#
# def random_flip(img, label):
# 	k = random.randint(0,5)
# 		label = label.transpose(Image.FLIP_LEFT_RIGHT)
# 	elif k == 2:
# 		img = img.transpose(Image.FLIP_TOP_BOTTOM)
# 		label = label.transpose(Image.FLIP_TOP_BOTTOM)
# 	elif k == 3:
# 		img = img.transpose(Image.ROTATE_90)
# 		label = label.transpose(Image.ROTATE_90)
# 	elif k == 4:
# 		img = img.transpose(Image.ROTATE_180)
# 		label = label.transpose(Image.ROTATE_180)
# 	elif k == 5:
# 		img = img.transpose(Image.ROTATE_270)
# 		label = label.transpose(Image.ROTATE_270)
# 	return img, label
#
# class MyDataset(Dataset):
#     def __init__(self, root_path, transform=None, target_transform=None):
#         self.img_pairs = load_image_path(root_path)
#         self.transform = transform
#         self.target_transform = target_transform
#
#     def __getitem__(self, index):
#         img_path, label_path = self.img_pairs[index]
#         img = change_image_channels(Image.open(img_path))
#         label = Image.open(label_path).convert('L')
#         img, label = random_crop(img, label, input_size)
#         # img, label = random_flip(img, label)
#         if self.transform is not None:
#         	img = self.transform(img)
#         if self.target_transform is not None:
#         	label = self.target_transform(label)
#         return img, label
#
#     def __len__(self):
#         return len(self.img_pairs)
#
# x_transform = transforms.Compose([
# 			transforms.ToTensor(),   #  =>[0,1]
# 			transforms.Normalize([0.5],[0.5])
# 	])
#
# target_transform = transforms.ToTensor()
#
# dataset = MyDataset(ROOT_PATH,x_transform,target_transform)
# data_loader = DataLoader(dataset, batch_size=4, shuffle=True)
#
# if __name__ == '__main__':
#     for i, (raw, label) in enumerate(data_loader):
#         # print(label[0][0])
#         cv2.imshow("img", raw[0][0].data.numpy())
#         cv2.imshow("label", label[0][0].data.numpy())
#         if cv2.waitKey()==27:break
#         # print(raw.shape)
#         # print(label.shape)




from torchvision.transforms import transforms
from torch.utils.data import Dataset, DataLoader
import PIL.Image as Image
import numpy as np
import os
import cv2
import random
from PIL import Image

ROOT_PATH = "data1/train"
input_size = [112, 112 ]


# input_size = [224, 224]

# def change_image_channels(image):
#     if image.mode == 'RGBA':
#         r, g, b, a = image.split()
#         image = Image.merge("RGB", (r, g, b))
#     img = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
#     # cv2.imshow("img1", img)
#     b, g, r = cv2.split(img)
#     clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
#     b = clahe.apply(b)
#     g = clahe.apply(g)
#     r = clahe.apply(r)
#     img = cv2.merge([b, g, r])
#     # cv2.imshow("img2", img)
#     # cv2.waitKey()
#     return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
def change_image_channels(image):
    if image.mode == 'RGBA':
        r, g, b, a = image.split()
        image = Image.merge("RGB", (r, g, b))
    img = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    # cv2.imshow("img1", img)
    b, g, r = cv2.split(img)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    b = clahe.apply(b)
    g = clahe.apply(g)
    r = clahe.apply(r)
    img = cv2.merge([b, g, r])
    # cv2.imshow("img2", img)
    # cv2.waitKey()
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
def change_image_channels_noclahe(image):
    if image.mode == 'RGBA':
        r, g, b, a = image.split()
        image = Image.merge("RGB", (r, g, b))
    img = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    # cv2.imshow("img1", img)
    b, g, r = cv2.split(img)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    b = clahe.apply(b)
    g = clahe.apply(g)
    r = clahe.apply(r)
    # img = cv2.merge([b, g, r])
    # cv2.imshow("img2", img)
    # cv2.waitKey()
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
def load_image_path(root_path):
    img_pairs = []
    raw_list = os.listdir(root_path + "/image")
    label_list = os.listdir(root_path + "/label_black")
    for raw_name, label_name in zip(raw_list, label_list):
        img_path = os.path.join(root_path, "image", raw_name)
        label_path = os.path.join(root_path, "label_black", label_name)
        img_pairs.append([img_path, label_path])
    return img_pairs


def random_crop(img, label, crop_size):
    size = img.size
    x_start = random.randint(0, size[0] - crop_size[0])
    y_start = random.randint(0, size[1] - crop_size[1])
    img = img.crop([x_start, y_start, x_start + crop_size[0], y_start + crop_size[1]])
    label = label.crop([x_start, y_start, x_start + crop_size[0], y_start + crop_size[1]])
    return img, label


def random_flip(img, label):
    k = random.randint(0, 5)
    if k == 1:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
        label = label.transpose(Image.FLIP_LEFT_RIGHT)
    elif k == 2:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
        label = label.transpose(Image.FLIP_TOP_BOTTOM)
    elif k == 3:
        img = img.transpose(Image.ROTATE_90)
        label = label.transpose(Image.ROTATE_90)
    elif k == 4:
        img = img.transpose(Image.ROTATE_180)
        label = label.transpose(Image.ROTATE_180)
    elif k == 5:
        img = img.transpose(Image.ROTATE_270)
        label = label.transpose(Image.ROTATE_270)
    return img, label


class MyDataset(Dataset):
    def __init__(self, root_path, transform=None, target_transform=None):
        self.img_pairs = load_image_path(root_path)
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img_path, label_path = self.img_pairs[index]
        img = change_image_channels(Image.open(img_path))
        # img = change_image_channels_noclahe(Image.open(img_path))
        label = Image.open(label_path).convert('L')
        img, label = random_crop(img, label, input_size)
        img, label = random_flip(img, label)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            label = self.target_transform(label)
        return img, label

    def __len__(self):
        return len(self.img_pairs)


x_transform = transforms.Compose([
    transforms.ToTensor(),  # =>[0,1]
    transforms.Normalize([0.5], [0.5])
])

target_transform = transforms.ToTensor()

dataset = MyDataset(ROOT_PATH, x_transform, target_transform)
data_loader = DataLoader(dataset, batch_size=12, shuffle=True)

if __name__ == '__main__':
    for i, (raw, label) in enumerate(data_loader):
        #     # print(label[0][0])
        cv2.imshow("img", raw[0][0].data.numpy())
        cv2.imshow("label", label[0][0].data.numpy())
        if cv2.waitKey() == 27: break
        # print(raw.shape)
        # print(label.shape)



