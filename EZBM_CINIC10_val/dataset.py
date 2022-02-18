# coding:utf-8
import torch
import torchvision
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
import os
import cv2
import imageio
import pickle
from PIL import Image
import glob


class CINIC10(data.Dataset):
    cls_num = 10

    def __init__(self, root, imb_type='exp', imb_factor=0.01, rand_number=0,
                 type='train', transform=None):
        super(CINIC10, self).__init__()
        self.rand_number = rand_number
        np.random.seed(rand_number)
        self.transform = transform

        file_path = os.path.join(root, type)
        self.imgs = np.array(glob.glob(file_path + '/*/*.png'))
        self.targets = []
        self.targets_dict = {}

        c = 0
        for i in range(self.imgs.shape[0]):
            img_sp = self.imgs[i].split('\\')[-2] # corresponding class
            if img_sp in self.targets_dict:
                self.targets.append(self.targets_dict[img_sp])
            else:
                self.targets.append(c)
                self.targets_dict[img_sp] = c
                c += 1

        if type == 'train':
            img_num_per_cls = self.get_img_num_per_cls(self.cls_num, imb_type, imb_factor)
            self.gen_imbalanced_data(img_num_per_cls)

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        # img_count = Counter(self.targets)
        # img_max = max(img_count, key=lambda x:img_count[x]) # 查找样本数最大的类
        img_max = int((len(self.imgs) / cls_num)*0.8)
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor ** (cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        return img_num_per_cls

    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list

    def gen_imbalanced_data(self, img_num_per_cls):
        new_imgs = []
        new_targets = []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)
        # np.random.shuffle(classes)
        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.seed(self.rand_number)
            np.random.shuffle(idx)
            selected_idx = idx[:the_img_num]
            new_imgs.extend(self.imgs[selected_idx])
            new_targets.extend([the_class, ] * the_img_num)

        self.imgs = new_imgs
        self.targets = new_targets

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, item):
        img = Image.open(self.imgs[item]).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        return img, self.targets[item]


class VALIDCINIC10(CINIC10):

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):
        # img_count = Counter(self.targets)
        # img_max = max(img_count, key=lambda x:img_count[x]) # 查找样本数最大的类
        img_max = int((len(self.imgs) / cls_num)*0.8)
        img_num_per_cls = []
        for cls_idx in range(cls_num):
            img_num_per_cls.append(img_max)

        return img_num_per_cls

    def gen_imbalanced_data(self, img_num_per_cls):
        new_imgs = []
        new_targets = []
        targets_np = np.array(self.targets, dtype=np.int64)
        classes = np.unique(targets_np)
        self.num_each_class = int(len(self.targets) / self.cls_num)
        # np.random.shuffle(classes)
        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            np.random.seed(self.rand_number)
            np.random.shuffle(idx)
            selected_idx = idx[the_img_num:]
            new_imgs.extend(self.imgs[selected_idx])
            new_targets.extend([the_class, ] * (self.num_each_class - the_img_num))

        self.imgs = new_imgs
        self.targets = new_targets


if __name__ == '__main__':
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    # trainset = IMBALANCECIFAR100(root='./data', train=True, download=True, transform=transform)

    # get cinic-10 data set
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.47889522, 0.47227842, 0.43047404], [0.24205776, 0.23828046, 0.25874835]),
    ]) # Normalize -> image=(image-mean)/std

    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.47889522, 0.47227842, 0.43047404], [0.24205776, 0.23828046, 0.25874835]),
    ])
    type = 'exp'
    factor = 0.01
    trainset = CINIC10(root='../../DataSet/cinic10', imb_type=type, imb_factor=factor,
                       rand_number=0, type='train', transform=transform_train)
    trainloader = iter(trainset)
    data, label = next(trainloader)
    '''以下做法没法读取, AttributeError: Can't get attribute 'CINIC10'''
    # saved_name_trn = 'cinic10' + '_' + type + '_' + str(factor) + '.pkl'
    # with open(saved_name_trn, 'wb') as fw:
    #     pickle.dump(trainset, fw)

    valset = VALIDCINIC10(root='../../DataSet/cinic10', type='train', transform=transform_val)
    valloader = iter(valset)
    valdata, vallabel = next(valloader)
    # saved_name_tst = 'cinic10' + '_' + 'test' + '.pkl'
    # with open(saved_name_tst, 'wb') as fw:
    #     pickle.dump(valset, fw)

    import pdb; pdb.set_trace()