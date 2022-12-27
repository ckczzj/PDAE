from PIL import Image
from io import BytesIO
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as F

from utils import open_lmdb

class CropCelebA64(object):
    def __call__(self, img):
        new_img = F.crop(img, 57, 25, 128, 128)
        return new_img

class CELEBA64(Dataset):
    def __init__(self, config, split="train", augmentation=True):
        super().__init__()
        self.config = config
        self.data_path = self.config["data_path"]
        self.image_channel = self.config["image_channel"]
        self.image_size = self.config["image_size"]
        self.split = split
        self.augmentation = augmentation

        if self.augmentation:
            self.transform = transforms.Compose([
                CropCelebA64(),
                transforms.Resize(self.image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,), inplace=True)
            ])
        else:
            self.transform = transforms.Compose([
                CropCelebA64(),
                transforms.Resize(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,), inplace=True)
            ])

    # train: 0~162,769  162,770
    # valid: 162,770~182,636   19,867
    # test: 182,637~202,599   19,963
    def __len__(self):
        if self.split == "train":
            return 162770
        if self.split == "valid":
            return 19867
        if self.split == "test":
            return 19963
        else:
            raise NotImplementedError()

    def __getitem__(self, index):
        if not hasattr(self, 'txn'):
            self.txn = open_lmdb(self.data_path)

        if self.split == "train":
            offset_index = index
        elif self.split == "valid":
            offset_index = 162770 + index
        elif self.split == "test":
            offset_index = 162770 + 19867 + index
        else:
            raise NotImplementedError()

        key = f'None-{str(offset_index).zfill(7)}'.encode('utf-8')
        img_bytes = self.txn.get(key)

        buffer = BytesIO(img_bytes)
        image = Image.open(buffer)

        image = self.transform(image)
        gt = image.mul(0.5).add(0.5).mul(255).add(0.5).clamp(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()

        return {
            "x_0": image,
            "gt": gt,
            "x_T": torch.randn(self.image_channel, self.image_size, self.image_size),
        }

    @staticmethod
    def collate_fn(batch):
        batch_size = len(batch)

        x_0=[]
        gts = []
        x_T=[]
        for i in range(batch_size):
            x_0.append(batch[i]["x_0"])
            gts.append(batch[i]["gt"])
            x_T.append(batch[i]["x_T"])

        x_0 = torch.stack(x_0, dim=0)
        x_T = torch.stack(x_T, dim=0)

        return {
            "net_input": {
                "x_0": x_0,
                "x_T": x_T,
            },
            "gts": np.asarray(gts),
        }
