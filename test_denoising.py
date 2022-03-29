import numpy as np
import os
import argparse
from tqdm import tqdm

import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

import scipy.io as sio
import cv2
from models import *
from dataloaders.data import get_validation_data, get_training_data
import utils
from skimage import img_as_ubyte
import scipy.io as scio
import h5py

parser = argparse.ArgumentParser(description='HSI denoising evaluation')
parser.add_argument('--input_dir', default='',
    type=str, help='Directory of validation images')
parser.add_argument('--ratio', default=50, type=int, help='Ratio')
parser.add_argument('--result_dir', default='',
    type=str, help='Directory for results')
parser.add_argument('--weights', default='',
    type=str, help='Path to weights')
parser.add_argument('--gpus', default='1', type=str, help='CUDA_VISIBLE_DEVICES')
parser.add_argument('--bs', default=1, type=int, help='Batch size for dataloader')
parser.add_argument('--save_images', action='store_true', help='Save denoised images in result directory')

args = parser.parse_args()


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

utils.mkdir(args.result_dir)

test_dataset = get_validation_data(args.input_dir, args.ratio)
test_loader = DataLoader(dataset=test_dataset, batch_size=args.bs, shuffle=False, num_workers=8, drop_last=False)

model_restoration = U_Net_3D()

utils.load_checkpoint(model_restoration,args.weights)
print("===>Testing using weights: ", args.weights)

model_restoration.cuda()

model_restoration=nn.DataParallel(model_restoration)

model_restoration.eval()


with torch.no_grad():
    for ii, data_test in enumerate(tqdm(test_loader), 0):
        hsi_gt = data_test[0].cuda()
        hsi_noisy = data_test[1].cuda()
        filenames = data_test[2]
        hsi_restored = model_restoration(hsi_noisy.unsqueeze(1))[:,0]
        hsi_restored = torch.clamp(hsi_restored,0,1)
     
        hsi_gt = hsi_gt.cpu().detach().numpy()
        hsi_noisy = hsi_noisy.cpu().detach().numpy()
        hsi_restored = hsi_restored.cpu().detach().numpy()

        if args.save_images:
            for batch in range(len(hsi_gt)):
                denoised_img = hsi_restored[batch, 30, ::-1]
                # cv2.imwrite(args.result_dir + filenames[batch][:-4] + '.png', np.rot90(denoised_img)*255)
                denoised_hsi = np.rot90(hsi_restored[batch, :, ::-1], axes=(-2,-1))
                sio.savemat(args.result_dir + filenames[batch][:-4] + '.mat', {'R_hsi': np.transpose(denoised_hsi, (1,2,0))})
        

