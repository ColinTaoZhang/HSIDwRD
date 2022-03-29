import numpy as np
import os
from torch.utils.data import Dataset
import torch
from utils import is_tif_file, load_tif_img, Augment_RGB_torch
import torch.nn.functional as F
import random
import scipy.stats as stats

augment   = Augment_RGB_torch()
transforms_aug = [method for method in dir(augment) if callable(getattr(augment, method)) if not method.startswith('_')] 

##################################################################################################
class DataLoaderTrain(Dataset):
    def __init__(self, data_dir, ratio=50, img_options=None, target_transform=None):
        super(DataLoaderTrain, self).__init__()

        self.target_transform = target_transform

        clean_files = sorted(os.listdir(os.path.join(data_dir, 'gt')))
        noisy_files = sorted(os.listdir(os.path.join(data_dir, 'input{}'.format(ratio))))
        
        self.clean_filenames = [os.path.join(data_dir, 'gt', x)          for x in clean_files if is_tif_file(x)]
        self.noisy_filenames = [os.path.join(data_dir, 'input{}'.format(ratio), x)     for x in noisy_files if is_tif_file(x)]

        self.clean = [torch.from_numpy(np.float32(load_tif_img(self.clean_filenames[index]))) for index in range(len(self.clean_filenames))]
        self.noisy = [torch.from_numpy(np.float32(load_tif_img(self.noisy_filenames[index]))) for index in range(len(self.noisy_filenames))]
        
        self.img_options=img_options

        self.tar_size = len(self.clean_filenames)  # get the size of target
        self.ratio = ratio

    def __len__(self):
        return self.tar_size

    def __getitem__(self, index):
        tar_index   = index % self.tar_size
        clean = self.clean[tar_index]
        noisy = self.noisy[tar_index]

        clean_filename = os.path.split(self.clean_filenames[tar_index])[-1]
        noisy_filename = os.path.split(self.noisy_filenames[tar_index])[-1]

        #Crop Input and Target
        ps = self.img_options['patch_size']
        H = clean.shape[1]
        W = clean.shape[2]
        r = np.random.randint(0, H - ps)
        c = np.random.randint(0, W - ps)
        clean = clean[:, r:r + ps, c:c + ps]
        noisy = noisy[:, r:r + ps, c:c + ps] * self.ratio

        apply_trans = transforms_aug[random.getrandbits(3)]

        clean = getattr(augment, apply_trans)(clean)
        noisy = getattr(augment, apply_trans)(noisy)        

        return clean, noisy, clean_filename, noisy_filename


##################################################################################################
class DataLoaderVal(Dataset):
    def __init__(self, data_dir, ratio=50, target_transform=None):
        super(DataLoaderVal, self).__init__()

        self.target_transform = target_transform

        clean_files = sorted(os.listdir(os.path.join(data_dir, 'gt')))
        noisy_files = sorted(os.listdir(os.path.join(data_dir, 'input{}'.format(ratio))))

        self.clean_filenames = [os.path.join(data_dir, 'gt', x)      for x in clean_files if is_tif_file(x)]
        self.noisy_filenames = [os.path.join(data_dir, 'input{}'.format(ratio), x) for x in noisy_files if is_tif_file(x)]

        self.clean = [torch.from_numpy(np.float32(load_tif_img(self.clean_filenames[index]))) for index in range(len(self.clean_filenames))]
        self.noisy = [torch.from_numpy(np.float32(load_tif_img(self.noisy_filenames[index]))) for index in range(len(self.noisy_filenames))]
        

        self.tar_size = len(self.clean_filenames)  
        self.ratio = ratio

    def __len__(self):
        return self.tar_size

    def __getitem__(self, index):
        tar_index   = index % self.tar_size

        clean = self.clean[tar_index]
        noisy = self.noisy[tar_index]
                
        clean_filename = os.path.split(self.clean_filenames[tar_index])[-1]
        noisy_filename = os.path.split(self.noisy_filenames[tar_index])[-1]

        val_or_not = True
        if val_or_not:
            ps = 128
            r = clean.shape[1]//2-ps//2
            c = clean.shape[2]//2-ps//2
            clean = clean[:, r:r + ps, c:c + ps]
            noisy = noisy[:, r:r + ps, c:c + ps] * self.ratio
        else:
            h = clean.shape[1]//16*16
            w = clean.shape[2]//16*16
            clean = clean[:, :h, :w]
            noisy = noisy[:, :h, :w] * self.ratio

        return clean, noisy, clean_filename, noisy_filename