import os
import random
import numpy as np
import cv2
from PIL import Image
import torch

from torch.utils.data import Dataset
from utils import hwc_to_chw, read_img


def augment(imgs=[], size=256, edge_decay=0., data_augment=True):
    H, W, _ = imgs[0].shape
    Hc, Wc = [size, size]

    # simple re-weight for the edge
    if random.random() < Hc / H * edge_decay:
        Hs = 0 if random.randint(0, 1) == 0 else H - Hc
    else:
        Hs = random.randint(0, H - Hc)

    if random.random() < Wc / W * edge_decay:
        Ws = 0 if random.randint(0, 1) == 0 else W - Wc
    else:
        Ws = random.randint(0, W - Wc)

    for i in range(len(imgs)):
        imgs[i] = imgs[i][Hs:(Hs + Hc), Ws:(Ws + Wc), :]

    if data_augment:
        # horizontal flip
        if random.randint(0, 1) == 1:
            for i in range(len(imgs)):
                imgs[i] = np.flip(imgs[i], axis=1)

        # bad data augmentations for outdoor dehazing
        rot_deg = random.randint(0, 3)
        for i in range(len(imgs)):
            imgs[i] = np.rot90(imgs[i], rot_deg, (0, 1))

    return imgs


def align(imgs=[], size=256):
    H, W, _ = imgs[0].shape
    Hc, Wc = [size, size]

    Hs = (H - Hc) // 2
    Ws = (W - Wc) // 2
    for i in range(len(imgs)):
        imgs[i] = imgs[i][Hs:(Hs + Hc), Ws:(Ws + Wc), :]

    return imgs


class RandomVerticalFlip(object):
    """Randomly vertically flips the given PIL.Image with a probability of 0.5  up-down"""
    def __call__(self, x, y):
        if random.random() < 0.5:
            x = np.copy(np.flipud(x))
            y = np.copy(np.flipud(y))
            x, y = Image.fromarray(x), Image.fromarray(y)
        return x, y


class RandomHorizontalFlip(object):
    """Randomly horizontally flips the given PIL.Image with a probability of 0.5 left-right"""
    def __call__(self, x, y):
        if random.random() < 0.5:
            '''Change the order of 0 and 1, for keeping the net search direction'''
            x = np.copy(np.fliplr(x))
            y = np.copy(np.fliplr(y))
            x, y = Image.fromarray(x), Image.fromarray(y)
        return x, y


class RandomGaussianNoise(object):
    def __init__(self, gaussian_para):
        self.mu = gaussian_para[0]
        self.std_var = gaussian_para[1]

    def __call__(self, x, y):
        x, y = np.array(x), np.array(y)
        shape = x.shape
        gaussian_noise = np.random.normal(self.mu, self.std_var, shape)
        x, y = x + (gaussian_noise*255), y + (gaussian_noise*255)
        x, y = x.clip(0, 255), y.clip(0, 255)
        x, y = x.astype(np.uint8), y.astype(np.uint8)
        x, y = Image.fromarray(x), Image.fromarray(y)
        return x, y


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std  = std
    def __call__(self, x, y):
        x = (np.array(x)/255.-self.mean)/self.std
        y = (np.array(y)/255.-self.mean)/self.std
        return x, y


class ToTensor(object):
    """Converts a numpy.ndarray (H x W x C) to a torch.FloatTensor of shape (C x H x W)."""
    def __call__(self, x, y):
        x = x * 2 - 1
        y = y * 2 - 1
        x = np.transpose(x, (2, 0, 1))
        y = np.transpose(y, (2, 0, 1))
        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()
        return x, y


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x, y):
        for t in self.transforms:
            x, y = t(x, y)
        return x, y


class PairLoader(Dataset):
    def __init__(self, root_dir, mode, size=256, edge_decay=0, data_augment=True, cache_memory=False):
        assert mode in ['train', 'valid', 'test']

        self.mode = mode
        self.size = size
        self.edge_decay = edge_decay
        self.data_augment = data_augment

        self.root_dir = root_dir
        self.img_names = sorted(os.listdir(os.path.join(self.root_dir, 'gt')))
        self.img_num = len(self.img_names)

        self.cache_memory = cache_memory
        self.source_files = {}
        self.target_files = {}
        self.t_values = {}

        # if self.mode == 'train':
        #     self.transform_list = [RandomVerticalFlip(),
        #                            RandomHorizontalFlip(),
        #                            RandomGaussianNoise([0, 1e-3]),
        #                            Normalize(0, 1),
        #                            ToTensor()
        #                            ]
        # else:
        #     self.transform_list = [
        #         Normalize(0, 1),
        #         ToTensor()
        #     ]
        # self.transform = Compose(self.transform_list)
        t_file_path = os.path.join(self.root_dir, 'input/haze_coef.txt')
        with open(t_file_path, 'r') as fin:
            for line in fin:
                idx_img, t_img = line[:-2].split(' ')
                self.t_values[idx_img] = t_img

    def __len__(self):
        return self.img_num

    def __getitem__(self, idx):
        cv2.setNumThreads(0)
        cv2.ocl.setUseOpenCL(False)

        # select a image pair
        img_name = self.img_names[idx]
        t_value = np.float32(self.t_values[img_name[:-4]])

        # read images
        if img_name not in self.source_files:
            source_img = read_img(os.path.join(self.root_dir, 'input', img_name), to_float=False)
            target_img = read_img(os.path.join(self.root_dir, 'gt', img_name), to_float=False)
            # source_img = Image.open(os.path.join(self.root_dir, 'HQ', img_name)).convert('RGB')
            # target_img = Image.open(os.path.join(self.root_dir, 'LQ_pre2', img_name)).convert('RGB')

            # cache in memory if specific (uint8 to save memory), need num_workers=0
            if self.cache_memory:
                self.source_files[img_name] = source_img
                self.target_files[img_name] = target_img
        else:
            # load cached images
            source_img = self.source_files[img_name]
            target_img = self.target_files[img_name]

        # target_numpy = np.asarray(target_img)
        gray_img = cv2.cvtColor(target_img, cv2.COLOR_RGB2GRAY)
        A_value = gray_img.astype('float32').mean() / 255.0
        # t_value = np.zeros_like(A_value)

        if self.mode == 'test':
            source_img = cv2.resize(source_img, (800, 800))
            target_img = cv2.resize(target_img, (800, 800))

        # [0, 1] to [-1, 1]
        source_img = source_img.astype('float32') / 255.0 * 2 - 1
        target_img = target_img.astype('float32') / 255.0 * 2 - 1

        # data augmentation
        if self.mode == 'train':
            [source_img, target_img] = augment([source_img, target_img], self.size, self.edge_decay, self.data_augment)
            # source_img, target_img = self.transform(source_img, target_img)

        if self.mode == 'valid':
            # [source_img, target_img] = align([source_img, target_img], 800)
            [source_img, target_img] = [source_img, target_img]
            # source_img, target_img = self.transform(source_img, target_img)

        return {'source': hwc_to_chw(source_img), 'target': hwc_to_chw(target_img), 'filename': img_name,
                't_value': t_value, 'A_value': A_value}


class SingleLoader(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.img_names = sorted(os.listdir(self.root_dir))
        self.img_num = len(self.img_names)

    def __len__(self):
        return self.img_num

    def __getitem__(self, idx):
        cv2.setNumThreads(0)
        cv2.ocl.setUseOpenCL(False)

        img_name = self.img_names[idx]
        img = read_img(os.path.join(self.root_dir, img_name)) * 2 - 1

        return {'img': hwc_to_chw(img), 'filename': img_name}
