from metrics import dice
import torch
import SimpleITK as sitk
from util.utils import Computer_half 
from util.utils import retu
from models import Unet
from models import vnet3d
from models import dskip_loc
import numpy as np
from segmentor_v1 import DenseNet
import config
import os
import h5py
from scipy import ndimage
args = config.getargs()
crop_size = args.crop_size
# checkpoint = './experments/'+ args.expername +'/best_model.pth'
checkpoint = './experments/zpc1/experments/3dskip12/best_model.pth'
name = './dataset/val.txt'
test_data_dir = './dataset/processed_data/process4/'
test_list = []
test_nii_list = []
with open(name) as f:
    for i in f:
        path = os.path.join(args.hdf5dir,i.rstrip('\n'))
        test_list.append(path)
test_nii_name = os.listdir('./dataset/test1/image/')
for i in test_nii_name:
    path = os.path.join('./dataset/test1/image/',i)
    test_nii_list.append(path)


# def get_set(nii,pred_array):
#     pred = sitk.GetImageFromArray(pred_array)
#     img_stk = sitk.ReadImage(nii)
#     resampler = sitk.ResampleImageFilter()
#     resampler.SetInterpolator(sitk.sitkLinear)#最近邻不引入新数值
#     resampler.SetOutputSpacing(img_stk.GetSpacing())
#     resampler.SetSize(img_stk.GetSize())
#     # resample on image
#     resampler.SetOutputOrigin(img_stk.GetOrigin())
#     resampler.SetOutputDirection(img_stk.GetDirection())
#     # print("Resampling image...")
#     pred_new = resampler.Execute(pred)
#     return pred_new
def get_set(nii,pred_array):
    # pred = sitk.GetImageFromArray(pred_array)
    img_stk = sitk.ReadImage(nii)
    target_size = sitk.GetArrayFromImage(img_stk).shape
    print(pred_array.shape)
    # resampler = sitk.ResampleImageFilter()
    pred = ndimage.zoom(pred_array,(target_size[0]/pred_array.shape[0],target_size[1]/pred_array.shape[1],target_size[2]/pred_array.shape[2]),order=0)
    print(pred.shape)
    pred = sitk.GetImageFromArray(pred)
    
    # pred.SetInterpolator(sitk.sitkLinear)#最近邻不引入新数值

    pred.SetSpacing(img_stk.GetSpacing())
    # pred.SetSize(img_stk.GetSize())
    # resample on image
    pred.SetOrigin(img_stk.GetOrigin())
    pred.SetDirection(img_stk.GetDirection())
    # print("Resampling image...")
    
    return pred
def read_med_image (file_path, dtype):
    img_stk = sitk.ReadImage(file_path)
    img_np = sitk.GetArrayFromImage(img_stk)
    img_np = img_np.astype(dtype)
    return img_np, img_stk



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = dskip_loc.DenseNet(num_init_features=32, growth_rate=16, block_config=(4, 4, 4, 4), drop_rate=0.2, num_classes=1).to(device)
# net = Unet.UNet(1, [16, 32, 48, 64, 96], 1, net_mode='3d').cuda()
if __name__ == '__main__':
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    test_list = []
    test_nii_list = []
    '''
    with open(name) as f:
        for i in f:
            image_path = os.path.join(args.hdf5dir,i.rstrip('\n'))
            test_list.append(image_path)
    '''
    test_nii_name = os.listdir('./dataset/test1/image/')
    name = test_nii_name[1:]
    print(test_list)
    print(test_nii_name)

    # test_nii_name = test_nii_name[1:]
    for i in name:
        path = os.path.join('./dataset/rawdata_r/image/',i)
        test_nii_list.append(path)
    # test_list = test_list[1:]
    # test_nii_list = test_nii_list[1:]
    print ('Checkpoint: ', checkpoint)
    saved_state_dict = torch.load(checkpoint)
    net.load_state_dict(saved_state_dict['net'])
    net.eval()
    # test_path = os.listdir('./dataset/test1/image/' )
    index_file = 0
    xstep = 64
    ystep = 64 # 16
    zstep = 64 # 16
    '''
    for i in range(len(test_list)):
        # _,image_stk = read_med_image(test_nii_list[i],dtype=np.float16)
        h5_file = h5py.File(test_list[i],'r')
        image_array = h5_file.get('data')[:].astype(np.float16)
        img = image_array.astype(np.float64)
    '''
    # for name in test_path:
    #     print(name)
    #     path = os.path.join('./dataset/test1/image/',name)
    #     inputs_T1, img_T1_itk = read_med_image(path , dtype=np.float32)
    #     mask = inputs_T1 > 0
    #     mask = mask.astype(bool)
    #     inputs_T1_norm = (inputs_T1 - inputs_T1[mask].mean()) / inputs_T1[mask].std()
    #     origin_shape = inputs_T1_norm.shape
    #     print(origin_shape)
    for i in name:
        image_array,_ = read_med_image('./dataset/process3/image/'+i,np.float32)
        image_array = image_array.astype(np.float32)
        mask  = image_array>0
        # Normalization
        image_array = ((image_array - image_array[mask].mean()) / image_array[mask].std()).astype(np.float16)
        print(image_array.shape)
        s,new_shape, image_array = Computer_half(image_array,(xstep,ystep,zstep),crop_size)
        print(s)
        
        image_array = image_array[:, :, :, None]
        inputs = image_array[None, :, :, :, :]
        image = inputs.transpose(0, 4, 1, 2, 3)
        image = torch.from_numpy(image).float().to(device)
        _, _, C, H, W = image.shape
        deep_slices   = np.arange(0, C - crop_size[0] + xstep, xstep)
        height_slices = np.arange(0, H - crop_size[1] + ystep, ystep)
        width_slices  = np.arange(0, W - crop_size[2] + zstep, zstep)
        whole_pred = np.zeros((1,)+(1,) + image.shape[2:])
        count_used = np.zeros((image.shape[2], image.shape[3], image.shape[4])) + 1e-5
        total = len(deep_slices) * len(height_slices) * len(width_slices)
        print('total-----',total)
        m = 0
        with torch.no_grad():
            for d in range(len(deep_slices)):
                for h in range(len(height_slices)):
                    for w in range(len(width_slices)):
                        deep = deep_slices[d]
                        height = height_slices[h]
                        width = width_slices[w]
                        image_crop = image[:, :, deep   : deep   + crop_size[0],
                                                    height : height + crop_size[1],
                                                    width  : width  + crop_size[2]]
                        outputs = torch.sigmoid(net(image_crop))
                        whole_pred[slice(None), slice(None), deep: deep + crop_size[0],
                                    height: height + crop_size[1],
                                    width: width + crop_size[2]] += outputs.data.cpu().numpy()
                        count_used[deep: deep + crop_size[0],
                                    height: height + crop_size[1],
                                    width: width + crop_size[2]] += 1
                        m = m+1
                        print('total---------',total,'now--------------------',m)
        whole_pred = whole_pred / count_used
        whole_pred = np.squeeze(whole_pred)
        print(whole_pred.max())
        print(type(whole_pred))
        print(whole_pred.shape)
        print(s)
        print(whole_pred.sum())
        whole_pred = retu(whole_pred,s)
        whole_pred[whole_pred>0.5]  = 1
        whole_pred[whole_pred<=0.5] = 0
        print(whole_pred.max())
        print(whole_pred.sum())
        print(whole_pred.dtype)
        # f_pred = os.path.join('./experments/','unet7/','pred/',i.replace('ct','seg'))
        f_pred = './experments/zpc1/experments/3dskip12/pred/best/'
        # f_pred_img = os.path.join('./experments/','unet7/','pred/',name[i])
        # f_pred = './experments/unet7/pred/111.ct.nii.gz'
        tar = os.path.join('./dataset/rawdata_r/image/',i)
        pred = get_set(tar,whole_pred)
        # img = get_set(test_nii_list[i],img)
        # print(sitk.read)
        # print(whole_pred.shape)
        # whole_pred_itk = sitk.GetImageFromArray(whole_pred.astype(np.uint8))
        # whole_pred_itk.SetSpacing(img_T1_itk.GetSpacing())
        # whole_pred_itk.SetDirection(img_T1_itk.GetDirection())
        # pred = sitk.GetImageFromArray(whole_pred)
        
        sitk.WriteImage(pred, f_pred)
        # sitk.WriteImage(img, f_pred_img)
        # img_stk = sitk.ReadImage(whole_pred)
        img_np = sitk.GetArrayFromImage(pred)
        print(img_np.max())
        print('------------------------------save----------------------------------------------------------')

