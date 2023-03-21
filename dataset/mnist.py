import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset
from torchvision import transforms

from utils import get_one_hot

class MNIST(Dataset):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.image_size = self.config["image_size"]
        self.image_channel = self.config["image_channel"]
        self.data_path = self.config["data_path"]
        self.train = self.config["train"]

        transform = transforms.Compose([
            transforms.Resize(self.image_size),
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
            "x_T": torch.randn(self.image_channel, self.image_size, self.image_size),
            "label": torch.tensor(label),
            "caption": str(int(label))
        }

        return ret

    @staticmethod
    def collate_fn(batch):
        batch_size = len(batch)

        indices = []
        gts = []
        x_0 = []
        x_T = []
        label = []
        for i in range(batch_size):
            indices.append(batch[i]["index"])
            gts.append(batch[i]["gt"])
            x_0.append(batch[i]["x_0"])
            x_T.append(batch[i]["x_T"])
            label.append(batch[i]["label"])

        x_0 = torch.stack(x_0, dim=0)
        x_T = torch.stack(x_T, dim=0)
        label = torch.stack(label, dim=0)
        condition = get_one_hot(label, 10)


        return {
            "net_input": {
                "x_0": x_0,
                "x_T": x_T,
                "condition": condition,
            },
            "gts": np.asarray(gts),
            "label": label,
            "captions": [s["caption"] for s in batch]
        }
