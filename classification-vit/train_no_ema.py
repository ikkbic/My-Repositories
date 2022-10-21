from __future__ import print_function  # 这个是python当中让print都以python3的形式进行print，即把print视为函数
import argparse  # 使得我们能够手动输入命令行参数，就是让风格变得和Linux命令行差不多
import torch  # 以下这几行导入相关的pytorch包，有疑问的参考我写的 Pytorch打怪路（一）系列博文
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchnet import meter
from models.ResNet import *
import torchvision
from matplotlib import pyplot as plt
from models.ViT_model import *

# Training settings 就是在设置一些参数，每个都有默认值，输入python main.py -h可以获得相关帮助
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=16, metavar='N',
                    # batch_size参数，如果想改，如改成128可这么写：python main.py -batch_size=128
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=10, metavar='N',  # test_batch_size参数，
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=150, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,  # GPU参数，默认为False
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=185, metavar='N',  # 跑多少次batch进行一次日志记录
                    help='how many batches to wait before logging training status')

args = parser.parse_args()  # 这个是使用argparse模块时的必备行，将参数进行关联，详情用法请百度 argparse 即可
args.cuda = not args.no_cuda and torch.cuda.is_available()  # 这个是在确认是否使用gpu的参数,比如

torch.manual_seed(args.seed)  # 设置一个随机数种子，相关理论请自行百度或google，并不是pytorch特有的什么设置
if args.cuda:
    torch.cuda.manual_seed(args.seed)  # 这个是为GPU设置一个随机数种子

kwargs = {'num_workers': 0, 'pin_memory': True} if args.cuda else {}

def load_dataset(batch_size = 16):
    """
    The output of torchvision datasets are PILImage images of range [0, 1].
    Transform them to Tensors of normalized range [-1, 1]
    """
    path_1 = r"E:\\我的资料\\EMA_proj\\train_dataset"
    path_2 = r"E:\\我的资料\\EMA_proj\\test_dataset"
    transform1 = transforms.Compose([transforms.Grayscale(),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.Resize((256, 256)),
                                     transforms.ToTensor(), transforms.Normalize((0.5), (0.5))])
    transform = transforms.Compose([transforms.Grayscale(),
                                    transforms.Resize((256, 256)),
                                    transforms.ToTensor(),transforms.Normalize((0.5), (0.5))])# transforms.Normalize((0.5), (0.5))
    trainset = torchvision.datasets.ImageFolder(path_1, transform=transform1)
    testset = torchvision.datasets.ImageFolder(path_2, transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(testset, batch_size=8, shuffle=True)

    return trainloader, testloader

train_loader, test_loader = load_dataset(args.batch_size)

model = vit_base_patch16_224_in21k(3,False).cuda()

criterion = nn.CrossEntropyLoss()
if args.cuda:
    model.cuda()  # 判断是否调用GPU模式

# optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)  # 初始化优化器 model.train()
#
# optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

train_loss = []
train_iteration = []
iteration = 0

def train(epoch):  # 定义每个epoch的训练细节
    model.train()  # 设置为trainning模式

    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:  # 如果要调用GPU模式，就把数据转存到GPU
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)  # 把数据转换成Variable
        optimizer.zero_grad()  # 优化器梯度初始化为零
        output = model(data)  # 把数据输入网络并得到输出，即进行前向传播

        loss = criterion(output, target)  # 计算损失函数
        loss.backward()  # 反向传播梯度
        optimizer.step()  # 结束一次前传+反传之后，更新优化器参数
        if (batch_idx % args.log_interval == 0) and (batch_idx != 0):  # 准备打印相关信息，args.log_interval是最开头设置的好了的参数

            train_loss.append(loss.item())

            # \tAccuracy: {}/{} ({:.0f}%)\n   计算训练集精度，format前面加上这个
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                        100. * batch_idx / len(train_loader), loss.item(),
                        # correct, len(train_loader.dataset),
                        # 100. * correct / len(train_loader.dataset)
            ))


def test():
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    classnum = 3
    target_num = torch.zeros((1, classnum))
    predict_num = torch.zeros((1, classnum))
    acc_num = torch.zeros((1, classnum))

    for inputs, targets in test_loader:
        if args.cuda:
            inputs, targets = inputs.cuda(), targets.cuda()
        inputs, targets = Variable(inputs, volatile=True), Variable(targets)
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        # loss is variable , if add it(+=loss) directly, there will be a bigger ang bigger graph.
        test_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        pre_mask = torch.zeros(outputs.size()).scatter_(1, predicted.cpu().view(-1, 1), 1.)
        predict_num += pre_mask.sum(0)
        tar_mask = torch.zeros(outputs.size()).scatter_(1, targets.data.cpu().view(-1, 1), 1.)
        target_num += tar_mask.sum(0)
        acc_mask = pre_mask * tar_mask
        acc_num += acc_mask.sum(0)

    recall = acc_num / target_num
    precision = acc_num / predict_num
    F1 = 2 * recall * precision / (recall + precision)
    accuracy = acc_num.sum(1) / target_num.sum(1)
    # 精度调整
    recall = (recall.numpy()[0] * 100).round(3)
    precision = (precision.numpy()[0] * 100).round(3)
    F1 = (F1.numpy()[0] * 100).round(3)
    accuracy = (accuracy.numpy()[0] * 100).round(3)

    # 打印格式方便复制
    print('recall: ', recall.mean())
    print('precision: ', precision.mean())
    print('F1: ', F1.mean())
    print('accuracy: ', accuracy)

for epoch in range(1, args.epochs + 1):  # 以epoch为单位进行循环
    train(epoch)
    test()

# 画出train loss 曲线
# for i in range(len(train_loss)):
#     train_iteration.append(iteration)
#     iteration += 200
#
# plt.figure()
# plt.plot(train_iteration, train_loss)
# plt.xlabel('iteration')
# plt.ylabel('train_loss')
# plt.show()

# 保存train loss
# 保存train loss
file = open('adam.txt', 'w')
for i in range(len(train_loss)):
    s = str(train_loss[i]) + '\n'
    file.write(s)
file.close()