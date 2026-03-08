import torch, torchvision, shutil
import torch.utils.data as data
import numpy as np
from torchvision.datasets import CIFAR10
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional
from tqdm.auto import tqdm

import os
import os.path
import logging

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)


class CIFAR10_truncated(data.Dataset):

    def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None, download=False):

        self.root = root
        self.dataidxs = dataidxs
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.download = download

        self.data, self.target = self.__build_truncated_dataset__()

    def __build_truncated_dataset__(self):

        cifar_dataobj = CIFAR10(self.root, self.train, self.transform, self.target_transform, self.download)

        if self.train:
            data = cifar_dataobj.data
            target = np.array(cifar_dataobj.targets)
        else:
            data = cifar_dataobj.data
            target = np.array(cifar_dataobj.targets)

        if self.dataidxs is not None:
            data = data[self.dataidxs]
            target = target[self.dataidxs]

        return data, target

    def __getitem__(self, index):
        img, target = self.data[index], self.target[index]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.data)

def train_aug(img:torch.Tensor, seed, mask:torch.Tensor=None):
    """Deterministic train augmentations"""
    # Flip
    if torch.randint(0, 2, (), generator=seed).item():
        img = img.flip((-1,))
        if mask is not None: mask = mask.flip((-1,))
    # Crop
    i = torch.randint(0, img.shape[1]-224+1, (), generator=seed).item()
    j = torch.randint(0, img.shape[2]-224+1, (), generator=seed).item()
    img = functional.crop(img, i, j, 224, 224)
    if mask is not None: mask = functional.crop(mask, i, j, 224, 224)
    # Color jitter
    brightness = 0.8 + torch.rand((), generator=seed).item()*0.4
    img *= brightness
    contrast = 0.8 + torch.rand((), generator=seed).item()*0.4
    mean = torch.mean(img, (1,2), keepdims=True)
    img = (img - mean) * contrast + mean
    return img, mask

class ImageNet(data.Dataset):
    def __init__(self, path:str="/data/weezeltggv/thesis/imnetproc", partition:str="train", n_clients:int=4, 
                 n_classes:int=1000, originalpath:str="/data/weezeltggv/thesis/imagenet"):
        # Set random seed. Wait for global seed to be set, so that each worker gets different seed.
        self.diff_seed = None 
        self.same_seed = torch.Generator()
        self.same_seed.manual_seed(42)
        # Augmentations
        self.val_crop = v2.CenterCrop(224)
        self.normalize = v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], inplace=True)
        # Create split
        if not os.path.exists(path):
            self.repartition(originalpath, path)
        self.partition = partition
        # Get class names and limit to n_classes by skipping
        with open(os.path.join(path, "mapping.txt")) as f:
            classes = f.readlines()
        classes = classes[::len(classes)//n_classes]
        classes = {i:line.split()[0] for i, line in enumerate(classes)}
        # Assign sample paths to each client
        self.data = {c: [] for c in range(n_clients)}
        self.targets = {c: [] for c in range(n_clients)}
        client = 0
        for class_idx in range(n_classes):
            classname = classes[class_idx]
            # Only n_classes
            if classname in classes.values():
                dirname = os.path.join(path, partition, classname)
                filelist = os.listdir(dirname)
                self.data[client].extend([
                    os.path.join(dirname, file) for file in filelist
                ])
                self.targets[client].extend([class_idx]*len(filelist))
                client = (client+1) % n_clients
        self.prev_idx = float("inf")
        # Misc attributes
        self.n_clients = n_clients
        self.n_classes = n_classes

    def __len__(self):
        # Take minimum of client lengths (many papers focus on quantity imbalance, which we do not address)
        return min([len(files) for files in self.data.values()])*self.n_clients
    
    def __getitem__(self, idx):
        # Initialize seed with global seed
        if self.diff_seed is None:
            self.diff_seed = torch.Generator()
            self.diff_seed.manual_seed(torch.randint(0, int(1e6), ()).item())
        # Shuffle per client at each epoch
        if idx<self.prev_idx:
            for c in self.data.keys():
                sorter = iter(torch.randperm(len(self.data[c]), generator=self.same_seed).tolist())
                self.data[c].sort(key=lambda _: next(sorter))
        self.prev_idx = idx
        # Load datum
        c = idx % self.n_clients
        i = idx // self.n_clients
        label, img_path = self.targets[c][i], self.data[c][i]
        with open(img_path, "rb") as f:
            bytesdata = torch.frombuffer(f.read(), dtype=torch.uint8)
        img = torchvision.io.decode_image(bytesdata, mode="RGB").float() / 255.
        # Apply augmentations
        img = functional.resize(img, 256)
        if self.partition=="train": 
            img, _ = train_aug(img, seed=self.diff_seed)
        else: img = self.val_crop(img)
        self.normalize(img)
        # Change to HWC
        img = torch.permute(img, (1,2,0))
        return label, img
    
class ImageNet_truncated(ImageNet):
    def __init__(self, dataidxs, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataidxs = dataidxs
    def __getitem__(self, idx):
        idx = self.dataidxs[idx]
        return super().__getitem__(idx)