from torch.utils.data import Dataset
import torch
import torchvision.transforms as transforms
import data_generate.transformations as transfm

from PIL import Image
from tqdm import tqdm
from itertools import chain
import pandas as pd
import glob
import random
import time

from train.util import split_path

class FewShotImageDataset(Dataset):
    def __init__(self, task_list, supercls=True, img_lvl=1, transform=None, relabel=None, device=None,
                 cuda_img_tensor=True, verbose='dataset'):
        self.task_list = task_list
        self.supercls = supercls # true if data has superclasses
        self.img_lvl = img_lvl # num of level below task dirs where imgs are located
        self.transform = transform
        self._relabel = relabel # tuple (colname, [val1, val2, ...])
        self.device = device
        self.cuda_img_tensor = cuda_img_tensor
        self.verbose = verbose

        self.df = self.generate_task_df()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if self.cuda_img_tensor:
            image = self.df.loc[idx, 'cuda_tensor']
        else:
            image = Image.open(self.df.loc[idx, 'img_path'])
            # check if non-grayscale images are in RGB mode. There's one image in mini_imagenet that has 4 channels.
            # convert grayscale to rgb if image mode is already grayscale but no grayscale transformation in transform list
            if ((image.mode == '1' or image.mode == 'L')
                and sum(transfm.__class__.__name__ == 'Grayscale' for transfm in self.transform.transforms) == 0) \
                    or (image.mode not in ('RGB', '1', 'L')):
                image = image.convert(mode='RGB')
            if self.transform is not None:
                image = self.transform(image)

        label = torch.tensor(self.df.loc[idx, 'cls_lbl'], device=self.device)
        return image, label

    @property
    def relabel(self):
        return self._relabel

    @relabel.setter
    def relabel(self, labels):
        self._relabel = labels

    def generate_task_df(self):
        img_dict_list = []
        for idx, classdir in \
                (tqdm(enumerate(self.task_list), desc='Generating {}'.format(self.verbose),
                         total=len(self.task_list)) if self.verbose is not None
                else enumerate(self.task_list)):
            # list all img paths in this class
            img_path_list = glob.glob(classdir + '/*' * self.img_lvl)
            # print(classdir + '/*' * self.img_lvl)
            # print(img_path_list)

            for img_path in img_path_list:
                # split img path per folders
                path_split = split_path(img_path)
                # append gpu_tensor (after transformation)
                if self.cuda_img_tensor:
    
                    image = Image.open(img_path)
                    # check if non-grayscale images are in RGB mode. There's one image in mini_imagenet that has 4 channels.
                    # convert grayscale to rgb if image mode is already grayscale but no grayscale transformation in transform list
                    if ((image.mode == '1' or image.mode == 'L')
                        and sum(transf.__class__.__name__ == 'Grayscale' for transf in self.transform.transforms) == 0) \
                            or (image.mode not in ('RGB', '1', 'L')):
                        image = image.convert(mode='RGB')
                    if self.transform is not None:
                        image = self.transform(image)
    

                else:
                    image = None
                # append info dict to list
                img_dict_list.append({
                    'supercls': path_split[-3],
                    'cls_name': '{}.{}'.format(path_split[-3], path_split[-2]),
                    'cls_lbl': idx,
                    'img_path': img_path,
                    'cuda_tensor': image
                } if self.supercls else {
                    'cls_name': path_split[-2],
                    'cls_lbl': idx,
                    'img_path': img_path,
                    'cuda_tensor': image
                })
        df_task = pd.DataFrame(img_dict_list)
        return df_task

    def relbl_df(self):
        for ind, value in enumerate(self.relabel[1]):
            self.df.loc[self.df[self.relabel[0]] == value, 'cls_lbl'] = ind


class RotatedMNISTTaskDataset(Dataset):
    """Dataset wrapper for a single RotMNIST task with support+query images."""
    def __init__(self, support_x, support_y, query_x, query_y, transform=None, device=None, cuda_img_tensor=True):
        import torch
        self.transform = transform
        self.device = device
        self.cuda_img_tensor = cuda_img_tensor
        self._relabel = None

        sx = torch.tensor(support_x, dtype=torch.float32).view(-1, 1, 28, 28)
        qx = torch.tensor(query_x, dtype=torch.float32).view(-1, 1, 28, 28)
        sy = torch.tensor(support_y, dtype=torch.int64)
        qy = torch.tensor(query_y, dtype=torch.int64)

        images = torch.cat([sx, qx], dim=0)
        labels = torch.cat([sy, qy], dim=0)

        rows = []
        tensors = []
        to_pil = transforms.ToPILImage()
        for img, lbl in zip(images, labels):
            if self.transform is not None:
                img_t = self.transform(to_pil(img))
            else:
                img_t = img
            if self.cuda_img_tensor and self.device is not None:
                img_t = img_t.to(self.device)
            tensors.append(img_t)
            rows.append({
                'supercls': 'rotmnist_task',
                'cls_name': str(int(lbl.item())),
                'cls_lbl': int(lbl.item()),
                'img_path': None,
                'cuda_tensor': img_t
            })
        self.df = pd.DataFrame(rows)
        self.tensors = tensors

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        import torch
        img = self.tensors[idx]
        label = torch.tensor(self.df.loc[idx, 'cls_lbl'], device=self.device)
        return img, label

    @property
    def relabel(self):
        return self._relabel

    @relabel.setter
    def relabel(self, labels):
        self._relabel = labels

    def relbl_df(self):
        if self._relabel is None:
            return
        for ind, value in enumerate(self._relabel[1]):
            self.df.loc[self.df[self._relabel[0]] == value, 'cls_lbl'] = ind


class PermutedMNISTTaskDataset(Dataset):
    """Dataset wrapper for a single PermMNIST task (full x/y arrays)."""
    def __init__(self, x, y, transform=None, device=None, cuda_img_tensor=True):
        import torch
        self.transform = transform
        self.device = device
        self.cuda_img_tensor = cuda_img_tensor
        self._relabel = None

        x = torch.tensor(x, dtype=torch.float32).view(-1, 1, 28, 28)
        y = torch.tensor(y, dtype=torch.int64).view(-1)

        rows = []
        tensors = []
        to_pil = transforms.ToPILImage()
        for img, lbl in zip(x, y):
            if self.transform is not None:
                img_t = self.transform(to_pil(img))
            else:
                img_t = img
            if self.cuda_img_tensor and self.device is not None:
                img_t = img_t.to(self.device)
            tensors.append(img_t)
            rows.append({
                'supercls': 'permmnist_task',
                'cls_name': str(int(lbl.item())),
                'cls_lbl': int(lbl.item()),
                'img_path': None,
                'cuda_tensor': img_t
            })
        self.df = pd.DataFrame(rows)
        self.tensors = tensors

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        import torch
        img = self.tensors[idx]
        label = torch.tensor(self.df.loc[idx, 'cls_lbl'], device=self.device)
        return img, label

    @property
    def relabel(self):
        return self._relabel

    @relabel.setter
    def relabel(self, labels):
        self._relabel = labels

    def relbl_df(self):
        if self._relabel is None:
            return
        for ind, value in enumerate(self._relabel[1]):
            self.df.loc[self.df[self._relabel[0]] == value, 'cls_lbl'] = ind


def get_df_inds_per_col_value(df, col, shuffle=True):
    inds_per_val = []
    for colval in df[col].unique():
        inds = df.loc[df[col] == colval].index.tolist()
        if shuffle:
            random.shuffle(inds)
        inds_per_val.append((colval, inds))
    return inds_per_val

def split_traintest_inds_per_cls(indices_per_class, num_test_per_class, rtn_chained=True):
    test_inds_per_cls = []
    train_inds_per_cls = []

    for cls, inds in indices_per_class:
        random.shuffle(inds)
        test_inds_per_cls.append((cls, inds[:num_test_per_class]))
        train_inds_per_cls.append((cls, inds[num_test_per_class:]))

    if rtn_chained:
        train_inds_chained = list(chain.from_iterable(list(zip(*train_inds_per_cls))[1]))
        test_inds_chained = list(chain.from_iterable(list(zip(*test_inds_per_cls))[1]))

        random.shuffle(train_inds_chained)
        random.shuffle(test_inds_chained)
        return train_inds_per_cls, train_inds_chained, test_inds_per_cls, test_inds_chained
    else:
        return train_inds_per_cls, test_inds_per_cls


class RotatedMNISTDataset(Dataset):
    """Few-shot friendly Rotated MNIST loader for BOML episodes."""

    def __init__(self, pkl_path, split='train', transform=None, device=None,
                 cuda_img_tensor=True, verbose='rotmnist'):
        self.transform = transform
        self.device = device
        self.cuda_img_tensor = cuda_img_tensor
        self.verbose = verbose
        self._relabel = None

        import pickle
        with open(pkl_path, 'rb') as f:
            payload = pickle.load(f)

        split_map = {'train': 0, 'val': 1, 'test': 2}
        idx = split_map.get(split, 0)
        if not isinstance(payload, (list, tuple)) or len(payload) <= idx:
            raise ValueError('Unexpected RotMNIST pickle structure')
        part = payload[idx]

        import torch
        images, labels = [], []
        for xs, ys in part:
            images.append(torch.tensor(xs, dtype=torch.float32))
            labels.append(torch.tensor(ys, dtype=torch.int64))
        images = torch.cat(images, dim=0)  # [N, 784]
        labels = torch.cat(labels, dim=0)
        images = images.view(-1, 1, 28, 28)

        df_rows = []
        tensors = []
        to_pil = transforms.ToPILImage()
        for i in range(images.size(0)):
            img = images[i]
            lbl = int(labels[i].item())
            if self.transform is not None:
                img_t = self.transform(to_pil(img))
            else:
                img_t = img
            if self.cuda_img_tensor and self.device is not None:
                img_t = img_t.to(self.device)
            tensors.append(img_t)
            df_rows.append({
                'supercls': 'rotmnist',
                'cls_name': str(lbl),
                'cls_lbl': lbl,
                'img_path': None,
                'cuda_tensor': img_t
            })

        self.df = pd.DataFrame(df_rows)
        self.tensors = tensors

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img = self.tensors[idx]
        label = torch.tensor(self.df.loc[idx, 'cls_lbl'], device=self.device)
        return img, label

    @property
    def relabel(self):
        return self._relabel

    @relabel.setter
    def relabel(self, labels):
        self._relabel = labels

    def relbl_df(self):
        if self._relabel is None:
            return
        for ind, value in enumerate(self._relabel[1]):
            self.df.loc[self.df[self._relabel[0]] == value, 'cls_lbl'] = ind
