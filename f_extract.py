from __future__ import print_function
import argparse

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from dbpn import Net as DBPN
from dbpn_v1 import Net as DBPNLL
from dbpn_iterative import Net as DBPNITER
from drbpn import Net as DRBPN
from d_drbpn import Net as D_DRBPN
from data import get_eval_set
from functools import reduce
from PIL import Image, ImageOps
from scipy.misc import imsave
import scipy.io as sio
import time, glob, math
import cv2
import skimage.metrics
from torchvision.transforms import ToTensor, ToPILImage
# Training settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--upscale_factor', type=int, default=4, help="super resolution upscale factor")
parser.add_argument('--recursion', type=int, default=3, help='Recursions num')
parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size')
parser.add_argument('--gpu_mode', type=bool, default=True)
parser.add_argument('--self_ensemble', type=bool, default=False)
parser.add_argument('--chop_forward', type=bool, default=False)
parser.add_argument('--threads', type=int, default=1, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--gpus', default=1, type=int, help='number of gpu')
parser.add_argument('--input_dir', type=str, default='Input')
parser.add_argument('--output', default='Results', help='Location to save checkpoint models')
parser.add_argument('--test_dataset', type=str, default='Set5')
parser.add_argument('--model_type', type=str, default='D-DRBPN')
parser.add_argument('--residual', type=bool, default=True)
parser.add_argument('--model', default='DIV2K_train_HRD-DRBPN4_recursion3_epoch_399.pth', help='sr pretrained base model')
parser.add_argument('--save_folder', default='weights', help='Location to save checkpoint models')
parser.add_argument('--f_extract', default='./Input/Set5/LR_bicubic/4/butterflyx4.png', help='f_extract')
opt = parser.parse_args()

gpus_list=range(opt.gpus)
print(opt)

cuda = opt.gpu_mode
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

torch.manual_seed(opt.seed)
if cuda:
    torch.cuda.manual_seed(opt.seed)

print('===> Loading datasets')


print('===> Building model')
if opt.model_type == 'DBPNLL':
    model = DBPNLL(num_channels=3, base_filter=64,  feat = 256, num_stages=10, scale_factor=opt.upscale_factor) ###D-DBPN
elif opt.model_type == 'DBPN-RES-MR64-3':
    model = DBPNITER(num_channels=3, base_filter=64,  feat = 256, num_stages=3, scale_factor=opt.upscale_factor) ###D-DBPN
elif opt.model_type == 'DRBPN':
    model = DRBPN(num_channels=3, base_filter=64,  feat = 256, num_stages=opt.recursion, scale_factor=opt.upscale_factor)
elif opt.model_type == 'D-DRBPN':
    model = D_DRBPN(num_channels=3, base_filter=64,  feat = 256, num_stages=opt.recursion, scale_factor=opt.upscale_factor)
else:
    model = DBPN(num_channels=3, base_filter=64,  feat = 256, num_stages=7, scale_factor=opt.upscale_factor) ###D-DBPN
    
if cuda:
    model = torch.nn.DataParallel(model, device_ids=gpus_list)
model_name = os.path.join(opt.save_folder, opt.model_type, str(opt.upscale_factor), opt.model)
print(model_name)
model.load_state_dict(torch.load(model_name, map_location=lambda storage, loc: storage))
print('Pre-trained SR model is loaded.')

  

if cuda:
    model = model.cuda(gpus_list[0])

def eval():
    model.eval()
    input = img = Image.open(opt.f_extract).convert('RGB')

    bicubic = rescale_img(input, opt.upscale_factor)
    input=Variable(ToTensor()(input)).unsqueeze(0)
    bicubic=Variable(ToTensor()(bicubic)).unsqueeze(0)
    if cuda:
        input = input.cuda(gpus_list[0])
        bicubic = bicubic.cuda(gpus_list[0])
    t0 = time.time()
    with torch.no_grad():
        for j in range(opt.recursion):
            prediction = resnet_cifar(model.module,input,j)
            #prediction = prediction + bicubic
            save_img(prediction.cpu().data, str(j)+'.png')
    t1 = time.time()
    print("===> Processing: %s || Timer: %.4f sec." % (opt.f_extract, (t1 - t0)))
    


def load_img(filepath):
    if os.path.exists(model_name):
        img = Image.open(filepath).convert('RGB')
        print(opt.f_extract)
    #y, _, _ = img.split() 
        return img

def resnet_cifar(net, x, j):
    x = net.feat0(x)
    l = net.feat1(x)
    h=l  
    results = []
    for i in range(j):
        h1 = net.up1(l)
        l1 = net.down1(h1)
        h2 = net.up2(l1)
            
        concat_h = torch.cat((h2, h1),1)
        l = net.down2(concat_h)
        concat_l = torch.cat((l, l1),1)
        h = net.up3(concat_l)
    results.append(h)
    results.append(h)
    results.append(h)
    
    results = torch.cat(results,1)
    x = net.output_conv(results)

    return x

def rescale_img(img_in, scale):
    size_in = img_in.size
    new_size_in = tuple([int(x * scale) for x in size_in])
    #q
    img_in = img_in.resize(new_size_in, resample=Image.BICUBIC)
    return img_in

def save_img(img, img_name):
    print("shape")
    print(img.shape)
    save_img = img.squeeze().clamp(0, 1).numpy().transpose(1,2,0)
    # save img
    print(save_img.shape)
    save_dir=os.path.join(opt.output,opt.test_dataset,str(opt.upscale_factor))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    save_fn = save_dir +'/'+ img_name
    cv2.imwrite(save_fn, cv2.cvtColor(save_img*255, cv2.COLOR_BGR2GRAY),  [cv2.IMWRITE_PNG_COMPRESSION, 0])




##Eval Start!!!!
print("Feature Extract Start! S. Jupiter!")
eval()
