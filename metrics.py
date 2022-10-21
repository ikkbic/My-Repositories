from cProfile import label
from importlib.resources import read_binary
import numpy as np
import torch
import SimpleITK as sitk
import os
import pandas as pd
import config
args = config.getargs() 
def read_med_image (file_path, dtype):
    img_stk = sitk.ReadImage(file_path)
    img_np = sitk.GetArrayFromImage(img_stk)
    img_np = img_np.astype(dtype)
    return img_np, img_stk
def dice(im1, im2):
    #print(im1.shape)
    #print(im2.shape)
    #im1 = im1[:,np.newaxis,:,:,:,:]
    #sim1 = np.squeeze(im1,1)
    #im1=im1==tid
    #im2=im2==tid
    im1=np.asarray(im1).astype(np.bool)
    im2=np.asarray(im2).astype(np.bool)
    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")
    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)
    print(intersection.sum())
    print('pre---------------------------------------',im1.sum())
    print('tar---------------------------------------',im2.sum())
    smooth = 0.0001
    dsc=   (2. * intersection.sum()) / (im1.sum() + im2.sum() + smooth)
    
    print(dsc)
    return dsc


def TN(logits, targets):
    #logits,targets = tensortobool(logits[:, ind, :, :, :],targets[:, ind, :, :, :])
    #logits = np.squeeze(logits,1)
    logits=np.asarray(logits).astype(np.bool)
    targets=np.asarray(targets).astype(np.bool)
    nlogits = np.logical_not(logits)
    ntargets = np.logical_not(targets)
    tp = np.sum(np.logical_and(logits, targets).astype(int))
    tn = np.sum(np.logical_and(nlogits,ntargets).astype(int)) 
    fp = np.sum(np.logical_and(nlogits,targets).astype(int))
    fn = np.sum(np.logical_and(logits,ntargets).astype(int))
    return tp,tn,fp,fn

def accmetric(logits, targets):
    tp,tn,fp,fn=TN(logits, targets)
    dsc  = dice(logits,targets)
    #logits = logits.numpy()
    #targets = targets.numpy()
    s = 0.0001
    p = tp/(tp+fp+s)
    r = tp/(tp+fn+s)
    sp = tn/(fp+tn+s)
    f1 = (2*p*r)/(p+r+s)
    #acc = np.sum(np.logical_not((np.logical_xor(logits[:, ind, :, :, :],targets[:, ind, :, :, :]))).astype(int))
    acc = (tp+tn)/(tp+tn+fp+fn+s)
    #return p,r,f1,acc,sp
    jacc = dsc/(2-dsc)
    return acc,p,dsc,r,sp,jacc

if __name__ == '__main__':
    prepath = os.listdir('./experments/zpc1/experments/3dvnet14/pred/best/' )
    tarpath = os.listdir('./dataset/process2/label/')
    t = len(prepath)
    acc_list = []
    p_list = []
    dsc_list = []
    r_list = []
    sp_list = []
    jacc_list = []
    acc_t=0
    p_t=0
    dsc_t=0
    r_t=0
    sp_t=0
    jacc_t=0
    for name in prepath:
        pre = os.path.join('./experments/zpc1/experments/3dvnet14/pred/best/',name)
        tar = os.path.join('./dataset/process2/label/',name.replace('ct','seg'))
        pre,_ = read_med_image(pre , dtype=np.uint8)
        tar,_ = read_med_image(tar,dtype=np.uint8)
        acc,p,dsc,r,sp,jacc = accmetric(pre, tar)
        acc_list.append(acc)
        p_list.append(p)
        dsc_list.append(dsc)
        r_list.append(r)
        sp_list.append(sp)
        jacc_list.append(jacc)
        acc_t+=acc
        p_t+=p
        dsc_t+=dsc
        r_t+=r
        sp_t+=sp
        jacc_t+=jacc
        
    acc = acc_t/t
    p = p_t/t
    dsc = dsc_t/t
    r = r_t/t
    sp = sp_t/t
    jacc = jacc_t/t 
    dataframe = pd.DataFrame({'name':prepath,'acc':acc_list,'p':p_list,'dsc':dsc_list,'r':r_list,'sp':sp_list,'jacc':jacc_list})
    trager_path = os.path.join('./experments/zpc1/experments/3dvnet14/','best.csv')
    dataframe.to_csv(trager_path,sep=',',index = False)
    print('---acc=',acc,'---p=',p,'---dsc=',dsc,'---sen=',r,'---sp=',sp,'---jacc=',jacc)


