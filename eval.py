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

from scipy.misc import imsave
import scipy.io as sio
import time, glob, math
import cv2
import skimage.metrics

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
parser.add_argument('--model_type', type=str, default='DBPNLL')
parser.add_argument('--residual', type=bool, default=True)
parser.add_argument('--model', default='DBPNLL_x8.pth', help='sr pretrained base model')
parser.add_argument('--save_folder', default='weights', help='Location to save checkpoint models')
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
test_set = get_eval_set(os.path.join(opt.input_dir,opt.test_dataset, "LR_bicubic", str(opt.upscale_factor)), opt.upscale_factor)
testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.testBatchSize, shuffle=False)

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
#model_para_path='./weights/' + 
model_name = os.path.join(opt.save_folder, opt.model_type, str(opt.upscale_factor), opt.model)
print(model_name)
if os.path.exists(model_name):
        model.load_state_dict(torch.load(model_name, map_location=lambda storage, loc: storage))
        print('Pre-trained SR model is loaded.')

if cuda:
    model = model.cuda(gpus_list[0])

def eval():
    model.eval()
    
    for batch in testing_data_loader:
        with torch.no_grad():
            input, bicubic, name = Variable(batch[0]), Variable(batch[1]), batch[2]
        if cuda:
            input = input.cuda(gpus_list[0])
            bicubic = bicubic.cuda(gpus_list[0])

        t0 = time.time()
        if opt.chop_forward:
            with torch.no_grad():
                prediction = chop_forward(input, model, opt.upscale_factor)
        else:
            if opt.self_ensemble:
                with torch.no_grad():
                    prediction = x8_forward(input, model)
            else:
                with torch.no_grad():
                    prediction = model(input)
                
        if opt.residual:
            prediction = prediction + bicubic

        t1 = time.time()
        print("===> Processing: %s || Timer: %.4f sec." % (name[0], (t1 - t0)))
        save_img(prediction.cpu().data, name[0])
'''在RGB通道上求PSNR和SSIM
def loadImg():
    image_list_sr = sorted(glob.glob(r"./" + opt.output+"/" + opt.test_dataset + "/" + str(opt.upscale_factor) + "/*.*"))
    image_list_hr = sorted(glob.glob(r"./" + opt.input_dir+"/" + opt.test_dataset + "/HR" + "/*.*"))
    length = len(image_list_sr)
    print(*image_list_sr, sep='\n')
    print(*image_list_hr, sep='\n')    
    
    print("总共有%d图片待处理" % length)
    avg_psnr = 0
    avg_ssim = 0
    for i in range(length):
        t3=time.time()
        sr = cv2.imread(image_list_sr[i])
        hr = cv2.imread(image_list_hr[i])
        print(image_list_sr[i])
        print(sr.shape)
        print(image_list_hr[i])
        print(hr.shape)
        sr=cv2.resize(sr, hr.shape)
        _, name = os.path.split(image_list_sr[i])
        psnr = skimage.metrics.peak_signal_noise_ratio(sr,hr, data_range=255)
        avg_psnr += psnr
        ssim = skimage.metrics.structural_similarity(sr, hr, multichannel=True)
        avg_ssim += ssim
        t4 = time.time()
        print("===> Processing: %s || Timer: %.4f sec. || psnr: %.4f db || ssim: %.4f" % (name, (t4 - t3), psnr, ssim))
    print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr / length))
    print("===> Avg. SSIM: {:.4f}".format(avg_ssim / length))
'''
#计算y通道上的PSNR
def loadImg():
    image_list_sr = sorted(glob.glob(r"./" + opt.output+"/" + opt.test_dataset + "/" + str(opt.upscale_factor) + "/*.*"))
    image_list_hr = sorted(glob.glob(r"./" + opt.input_dir+"/" + opt.test_dataset + "/HR" + "/*.*"))
    length = len(image_list_sr)
    print(*image_list_sr, sep='\n')
    print(*image_list_hr, sep='\n')    
    shave_border=opt.upscale_factor
    print("总共有%d图片待处理" % length)
    avg_psnr = 0
    avg_ssim = 0
    for i in range(length):
        t3=time.time()
        sr = cv2.imread(image_list_sr[i])
        hr = cv2.imread(image_list_hr[i])
        sr = cv2.cvtColor(sr, cv2.COLOR_BGR2YCR_CB)
        sr = sr[:,:,0]
        hr = cv2.cvtColor(hr, cv2.COLOR_BGR2YCR_CB)
        hr = hr[:,:,0]
        print(image_list_sr[i])
        if (i==0):
            print(sr)
        print(sr.shape)
        print(image_list_hr[i])
        print(hr.shape)
        if (sr.shape!=hr.shape):
            print("图片大小不等")
            sr=cv2.resize(sr, (hr.shape[1], hr.shape[0]), interpolation = cv2.INTER_CUBIC)
            print(sr.shape)
            print(hr.shape)
        height, width = sr.shape
        hr = hr[shave_border:height - shave_border, shave_border:width - shave_border]
        sr = sr[shave_border:height - shave_border, shave_border:width - shave_border]
        print("截去边缘像素")
        print(sr.shape)
        print(hr.shape)
        _, name = os.path.split(image_list_sr[i])
        psnr = skimage.metrics.peak_signal_noise_ratio(sr,hr, data_range=255)
        avg_psnr += psnr
        ssim = skimage.metrics.structural_similarity(sr, hr)
        avg_ssim += ssim
        t4 = time.time()
        print("===> Processing: %s || Timer: %.4f sec. || psnr: %.4f db || ssim: %.4f" % (name, (t4 - t3), psnr, ssim))
    print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr / length))
    print("===> Avg. SSIM: {:.4f}".format(avg_ssim / length))

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
    cv2.imwrite(save_fn, cv2.cvtColor(save_img*255, cv2.COLOR_BGR2RGB),  [cv2.IMWRITE_PNG_COMPRESSION, 0])

def x8_forward(img, model, precision='single'):
    def _transform(v, op):
        if precision != 'single': v = v.float()

        v2np = v.data.cpu().numpy()
        if op == 'vflip':
            tfnp = v2np[:, :, :, ::-1].copy()
        elif op == 'hflip':
            tfnp = v2np[:, :, ::-1, :].copy()
        elif op == 'transpose':
            tfnp = v2np.transpose((0, 1, 3, 2)).copy()
        
        ret = torch.Tensor(tfnp).cuda()

        if precision == 'half':
            ret = ret.half()
        elif precision == 'double':
            ret = ret.double()

        with torch.no_grad():
            ret = Variable(ret)

        return ret

    inputlist = [img]
    for tf in 'vflip', 'hflip', 'transpose':
        inputlist.extend([_transform(t, tf) for t in inputlist])

    outputlist = [model(aug) for aug in inputlist]
    for i in range(len(outputlist)):
        if i > 3:
            outputlist[i] = _transform(outputlist[i], 'transpose')
        if i % 4 > 1:
            outputlist[i] = _transform(outputlist[i], 'hflip')
        if (i % 4) % 2 == 1:
            outputlist[i] = _transform(outputlist[i], 'vflip')
    
    output = reduce((lambda x, y: x + y), outputlist) / len(outputlist)

    return output
    
def chop_forward(x, model, scale, shave=8, min_size=80000, nGPUs=opt.gpus):
    b, c, h, w = x.size()
    h_half, w_half = h // 2, w // 2
    h_size, w_size = h_half + shave, w_half + shave
    inputlist = [
        x[:, :, 0:h_size, 0:w_size],
        x[:, :, 0:h_size, (w - w_size):w],
        x[:, :, (h - h_size):h, 0:w_size],
        x[:, :, (h - h_size):h, (w - w_size):w]]

    if w_size * h_size < min_size:
        outputlist = []
        for i in range(0, 4, nGPUs):
            with torch.no_grad():
                input_batch = torch.cat(inputlist[i:(i + nGPUs)], dim=0)
            if opt.self_ensemble:
                with torch.no_grad():
                    output_batch = x8_forward(input_batch, model)
            else:
                with torch.no_grad():
                    output_batch = model(input_batch)
            outputlist.extend(output_batch.chunk(nGPUs, dim=0))
    else:
        outputlist = [
            chop_forward(patch, model, scale, shave, min_size, nGPUs) \
            for patch in inputlist]

    h, w = scale * h, scale * w
    h_half, w_half = scale * h_half, scale * w_half
    h_size, w_size = scale * h_size, scale * w_size
    shave *= scale

    with torch.no_grad():
        output = Variable(x.data.new(b, c, h, w))

    output[:, :, 0:h_half, 0:w_half] \
        = outputlist[0][:, :, 0:h_half, 0:w_half]
    output[:, :, 0:h_half, w_half:w] \
        = outputlist[1][:, :, 0:h_half, (w_size - w + w_half):w_size]
    output[:, :, h_half:h, 0:w_half] \
        = outputlist[2][:, :, (h_size - h + h_half):h_size, 0:w_half]
    output[:, :, h_half:h, w_half:w] \
        = outputlist[3][:, :, (h_size - h + h_half):h_size, (w_size - w + w_half):w_size]

    return output

##Eval Start!!!!
print("Eval Start! S. Jupiter!")
eval()
print("Test Start! PSNR and SSIM.")
loadImg()