import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset
from torchvision import transforms

from utils.utils import get_one_hot

class MNIST(Dataset):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.image_size = self.config["image_size"]
        self.image_channel = self.config["image_channel"]
        self.data_path = self.config["data_path"]
        self.train = self.config["train"]

        transform = transforms.Compose([
            transforms.Resize((self.image_size,self.image_size)),
            transforms.ToTensor(),  # 0 ~ 1
            transforms.Normalize((0.5,), (0.5,), inplace=True),  # -1 ~ 1
        ])

        self.dataset = torchvision.datasets.MNIST(self.data_path, train=self.train, download=True, transform=transform)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        image, label = self.dataset.__getitem__(index)
        gt = image.mul(0.5).add(0.5).mul(255).add(0.5).clamp(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()

        ret = {
            "index": index,
            "gt": gt,
            "x_0": image,
            "label": torch.tensor(label),
            "caption": str(int(label))
        }

        return ret

    @staticmethod
    def collate_fn(batch):
        batch_size = len(batch)

        idx=[]
        x_0=[]
        gt=[]
        label = []

        for i in range(batch_size):
            idx.append(batch[i]["index"])
            x_0.append(batch[i]["x_0"])
            gt.append(batch[i]["gt"])
            label.append(batch[i]["label"])

        x_0 = torch.stack(x_0, dim=0)
        label = torch.stack(label, dim=0)
        condition = get_one_hot(label, 10)

        return {
            "idx": idx,
            "x_0": x_0,
            "gts": np.asarray(gt),
            "condition": condition,
            "label": label,
            "captions": [s["caption"] for s in batch]
        }
