from yacs.config import CfgNode as CN
import shutil
import os
import sys
_C = CN()
_C.num_epoch = 20000
# _C.lr_S = 2e-3
_C.lr_D = 2e-4
_C.momentum_S=0.9
_C.momentum_D=0.9
_C.step_size_S = 50
_C.step_size_D = 5000
_C.beta1=0.9
_C.beta2=0.999
_C.batch_train = 2
_C.loss_k = 0.5
_C.pre_trained=True

early_stop = 100
# -------------------------------------------------------------------------------------------
_C.n_threads = 2
_C.cpu = False
# 是否使用数据增强
_C.argument = True
# Preprocess parameters
_C.n_labels = 1
_C.upper = 1000
_C.lower = -1000
_C.xy_down_slice = 0.5
_C.z_down_slice = 1.0

# Dataset parameters
_C.train_dataset_path = './datasets/train/'
_C.val_dataset_path = './datasets/val/'
_C.test_dataset_path = './datasets/test/'
_C.save_path = 'Unet'

# train
_C.epochs = 200
_C.lr = 0.002
_C.early_stop = 100
_C.batch_size = 2
_C.expername = 'Default'
_C.default_config = False



_C.early_stop = 100

_C.data_dm = 2
_C.ignore_label = 9  
_C.num_classes= 1
_C.crop_size = (64, 128, 128)
_C.readme = ''

_C.checkpoint_name= 'model_3d_denseseg_v1'
# _C.note_S='Seg_3ddenseseg(Adam lr_S: ' + str(lr_S) + ',w_decay:1e-4' + 'beta:' +str(beta1)+ ',' + str(beta2) + ',' + 'step:' + str(step_size_S) + ' , lr_step)'
# _C.note_D='Seg_3ddenseseg(Adam lr_S: ' + str(lr_S) + ',w_decay:1e-4' + 'beta:' +str(beta1)+ ',' + str(beta2) + ',' + 'step:' + str(step_size_S) + ' , lr_step)'

# _C.num_checkpoint='00100'
# _C.expermentsname = 'unet'
_C.note= _C.expername
# #Testing
# check = './experments/'+_C.expername +'/best_model.pth'
_C.checkpoint= './experments/'+ _C.expername +'/best_model.pth'
_C.localtion = True
_C.hdf5dir = './dataset/processed_data/process4/'
#test

'''
_C.train_or_test         = 'train'
_C.output_dir            = 'logs/1027densenet'
_C.aug                   = True
_C.latest_checkpoint     = 'checkpoint_latest.pt'
_C.total_epochs          = 10000
_C.epochs_per_checkpoint = 10
_C.batch_size            = 2
_C.ckpt                  = None
_C.init_lr               = 0.002
_C.scheduer_step_size    = 20
_C.scheduer_gamma        = 0.8
_C.debug                 = False
_C.mode                  = '3d' # '2d or '3d'
_C.in_class              = 1
_C.out_class = 1
_C.crop_or_pad_size = 512,512,256 # if 2D: 256,256,1
_C.patch_size = 128,128,32 # if 2D: 128,128,1 
_C.patch_overlap = 4,4,4 # if 2D: 4,4,0
_C.fold_arch = '*.nii.gz'
_C.save_arch = '.nii.gz'
_C.source_train_dir = './datasets/cod1/train/image/'
_C.label_train_dir  = './datasets/cod1/train/label/'
_C.source_test_dir  = './datasets/cod1/test/image/'
_C.label_test_dir   = './datasets/cod1/test/label/'
_C.source_val_dir   = './datasets/cod1/val/image/'
_C.label_val_dir    = './datasets/cod1/val/label/'
_C.output_dir_test  = 'logs/1027densenet'
'''

def getconfig (cfg_file='./config.yaml'):
    config1 = _C.clone()
    if cfg_file is not None:
        config1.merge_from_file(cfg_file)
    return config1

# config = getconfig()
def getargs():
    config = getconfig()
    expername = config.expername
    use_default = config.default_config
    expath = './experments/'+str(expername)
    if not os.path.exists(expath): 
        os.mkdir(expath)
        os.mkdir(os.path.join(expath,'pred'))
    #config1 = getconfig1()
    config = _C.clone()
    if use_default ==True:
        config.merge_from_file('./experments/default_config.yaml')
        shutil.copy('./experments/default_config.yaml','./ experments/'+str(expername)+'/config.yaml')
    else:
        config.merge_from_file('./config.yaml')
        shutil.copy('./config.yaml','./experments/'+str(expername)+'/config.yaml')
    return config
