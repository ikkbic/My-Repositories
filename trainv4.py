import os
from collections import OrderedDict
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
#import config
#from models.unet import Unet
#from models.fcn import FCN32s
#from models.deeplabv3 import DeepLabV3
#from models.segnet import SegNet
#from models.highresnet import HighRes2DNet
#from utils.dataset import data_train,data_val
from util import logger, loss, metrics, weights_init
from util.loss import Focal_Dice_Loss
#from utils import logger
#import config
import os
import torch.utils.data as dataloader
from dataloader import H5Dataset
import torch.optim as optim
# from common import *
from util.foclaloss import FocalLoss
from util.diceloss import DiceLoss
from tqdm import tqdm
from shutil import copy
from torch.cuda.amp import autocast as autocast
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# exname = 'uuu'
from models import Unet
import config
from models import vnet3d
from models.transunet3d import transunet
from models.transunet3d import CONFIGS as CONFIGS_ViT_seg
import torch.cuda.amp as amp
from torch.utils.data import DataLoader
from prefetch_generator import BackgroundGenerator

class DataLoaderX(DataLoader):

    def __iter__(self):
        return BackgroundGenerator(super().__iter__())
vit = CONFIGS_ViT_seg['ViT-B_16'] 
# if not os.path.exists(save_path): os.mkdir(save_path)
# use_default_config = False

# if use_default_config == False:
#     config_path = None
#     copy('./experments/default_config.yaml',os.path.join(save_path,'config.yaml')) 
# else :
#     config_path = './config.yaml'
#     copy(config_path,os.path.join(save_path,'config.yaml'))




args = config.getargs() 
save_path = os.path.join('./experments', args.expername)
device = torch.device('cpu' if args.cpu else 'cuda')
save_path = os.path.join('./experments', args.expername)

scaler = amp.GradScaler()
size_sum = args.crop_size[0]*args.crop_size[1]*args.crop_size[2]
def val(model, val_loader, loss_func, metrics_func):
    model.eval()
    losss=0.0
    metrics=0.0
    with torch.no_grad():
        for idex, (data, target)in enumerate(val_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss   = loss_func(output, target)
            target_sum = torch.sum(target)
            # alpha = math.sin(math.tanh(3*target_sum/(2*size_sum*loss_k))*math.pi)
            metric = metrics_func(output, target)
            losss+=loss
            metrics+=metric
    loss = (losss/len(val_loader)).detach().cpu().numpy().round(5)
    metric = (metrics/len(val_loader)).round(5)
    val_log = OrderedDict({'Val_loss': loss, 'Val_accu': metric[0],'Val_precision':metric[1],
                           'Val_dice':metric[2],'Val_sensitivity':metric[3],'Val_specificity':metric[4],'Val_jacc':metric[5]})
    return val_log

def train(model, train_loader, optimizer, loss_func, metric_func):

    model.train()
    losss = 0.0
    metrics = 0.0
    
    for idex,(data, target) in enumerate(train_loader):

        optimizer.zero_grad()
        data, target = data.to(device), target.to(device)
        with autocast():
            output = model(data)
            loss = loss_func(output, target)
        print(loss)
        scaler.scale(loss).backward()
        # loss.backward()
        metric = metric_func(output, target)
        scaler.step(optimizer)
        # optimizer.step()
        scaler.update()
        losss +=loss
        metrics +=metric
    loss = (losss/len(train_loader)).detach().cpu().numpy().round(5)
    metric = (metrics/len(train_loader)).round(5)
    val_log = OrderedDict({'Train_loss': loss, 'Train_accu': metric[0],'Train_precision':metric[1],
                           'Train_dice':metric[2],'Train_sensitivity':metric[3],'Train_specificity':metric[4],'Train_jacc':metric[5]})
    return val_log

def main():
    # --------------------------CUDA check-----------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    # -------------init Seg---------------
    # model_S = DenseNet(num_init_features=32, growth_rate=16, block_config=(4, 4, 4, 4), drop_rate=0.2, num_classes=num_classes).to(device)
    model_S = Unet.UNet(1, [16, 32, 48, 64, 96], 1, net_mode='3d').cuda()
    # saved_state_dict = torch.load('./experments/unet5/latest_model.pth')
    # model_S.load_state_dict(saved_state_dict['net'])
    # model_S = vnet3d.VNet().cuda()
    # model_S = transunet(vit).cuda()
    # --------------Loss---------------------------
    #criterion_S = nn.CrossEntropyLoss().cuda()
    criterion_1 = FocalLoss().cuda()   
    criterion_2 = DiceLoss().cuda()
    # setup optimizer
    optimizer_S = optim.Adam(model_S.parameters(), lr=args.lr, weight_decay=6e-4, betas=(0.97, 0.999))
    scheduler_S = optim.lr_scheduler.StepLR(optimizer_S, step_size=args.step_size_S, gamma=0.2)
    mri_data_train = H5Dataset(args.train_dataset_path, mode='train')
    trainloader = DataLoaderX(mri_data_train, batch_size=args.batch_train, shuffle=True,num_workers=1 ,pin_memory=True)
    mri_data_val = H5Dataset(args.val_dataset_path, mode='val')
    valloader = DataLoaderX(mri_data_val, batch_size=4, shuffle=False,num_workers=1,pin_memory=True)
    print('Rate     | epoch  | Loss seg| DSC_val')

    #train_loader = DataLoader(dataset=data_train(args),batch_size=args.batch_size,num_workers=args.n_threads, shuffle=True)
    #val_loader = DataLoader(dataset=data_val(args),batch_size=1,num_workers=args.n_threads, shuffle=False)

    #model = Unet( 3, args.n_labels).to(device)
    #model = FCN32s(3, args.n_labels).to(device)
    #model = DeepLabV3(3, args.n_labels).to(device)
    #model.apply(weights_init.init_model)
    #model = SegNet(3, args.n_labels).to(device)
    #model = HighRes2DNet(3, args.n_labels).to(device)
    #optimizer = optim.Adam(model.parameters(), lr=args.lr)
    #model = torch.nn.DataParallel(model, device_ids=args.gpu_id)  # multi-GPU
    #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
    #loss = torch.nn.BCEWithLogitsLoss().to(device)
    loss = Focal_Dice_Loss().cuda()
    metric = metrics.metric

    log = logger.Train_Logger(save_path,"train_log")

    best = [0,0] # 初始化最优模型的epoch和performance
    trigger = 0  # early stop 计数器
    
    for epoch in range(1, args.num_epoch + 1):

        print("==========Epoch:{}==========lr:{}".format(epoch,optimizer_S.state_dict()['param_groups'][0]['lr']))


        # print()
        train_log = train(model_S, trainloader, optimizer_S, loss, metric)
        print('-----------------------------val-------------------------------------------')
        print('-----------------------------val-------------------------------------------')
        print('-----------------------------val-------------------------------------------')
        val_log = val(model_S, valloader, loss, metric)
        scheduler_S.step()
        log.update(epoch,train_log,val_log)
        print('train_loss: ',train_log['Train_loss'],'   val_loss: ',val_log['Val_loss'],
              '   train_dice: ',train_log['Train_dice'],'   val_dice: ',val_log['Val_dice'])
        state = {'net': model_S.state_dict(),'optimizer':optimizer_S.state_dict(),'epoch': epoch}
        torch.save(state, os.path.join(save_path, 'latest_model.pth'))
        trigger += 1
        if val_log['Val_dice'] > best[1]:
            print('Saving best model')
            torch.save(state, os.path.join(save_path, 'best_model.pth'))
            best[0] = epoch
            best[1] = val_log['Val_dice']
            trigger = 0
        print('Best performance at Epoch: {} | {}'.format(best[0],best[1]))

        if args.early_stop is not None:
            if trigger >= args.early_stop:
                print("=> early stopping")
                break
        torch.cuda.empty_cache()    


if __name__ == '__main__' :
    main()