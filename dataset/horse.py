import numpy as np
from PIL import Image
from io import BytesIO
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from utils.utils import open_lmdb

class HORSE(Dataset):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.image_size = self.config["image_size"]
        self.image_channel = self.config["image_channel"]
        self.data_path = self.config["data_path"]
        self.augmentation = self.config["augmentation"]

        if self.augmentation:
            self.transform = transforms.Compose([
                transforms.Resize((self.image_size,self.image_size)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((self.image_size,self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

    def __len__(self):
        return 2000340

    def __getitem__(self, index):
        if not hasattr(self, 'txn'):
            self.txn = open_lmdb(self.data_path)

        key = f'256-{str(index).zfill(7)}'.encode('utf-8')
        img_bytes = self.txn.get(key)

        buffer = BytesIO(img_bytes)
        image = Image.open(buffer)

        image = self.transform(image)
        gt = image.mul(0.5).add(0.5).mul(255).add(0.5).clamp(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()

        return {
            "idx": index,
            "x_0": image,
            "gt": gt,
        }

    @staticmethod
    def collate_fn(batch):
        batch_size = len(batch)

        idx=[]
        x_0=[]
        gt=[]

        for i in range(batch_size):
            idx.append(batch[i]["idx"])
            x_0.append(batch[i]["x_0"])
            gt.append(batch[i]["gt"])

        x_0 = torch.stack(x_0, dim=0)

        return {
            "idx": idx,
            "x_0": x_0,
            "gts": np.asarray(gt),
        }
