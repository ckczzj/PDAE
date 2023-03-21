import numpy as np
from PIL import Image
from io import BytesIO
import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from utils import open_lmdb


class CELEBAHQ(Dataset):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.data_path = self.config["data_path"]
        self.image_channel = self.config["image_channel"]
        self.image_size = self.config["image_size"]
        self.augmentation = self.config["augmentation"]

        if self.augmentation:
            self.transform = transforms.Compose([
                transforms.Resize(self.image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,), inplace=True)
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(self.image_size),
                transforms.ToTensor(),  # 0 ~ 1
                transforms.Normalize((0.5,), (0.5,), inplace=True)
            ])

        with open(os.path.join(self.data_path, "CelebAMask-HQ-attribute-anno.txt")) as f:
            f.readline() # discard the top line
            self.df = pd.read_csv(f, delim_whitespace=True)

        self.id_to_label = [
            '5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes',
            'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair',
            'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin',
            'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones',
            'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard',
            'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline',
            'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair',
            'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick',
            'Wearing_Necklace', 'Wearing_Necktie', 'Young'
        ]
        self.label_to_id = {v: k for k, v in enumerate(self.id_to_label)}

    def __len__(self):
        return 30000

    def __getitem__(self, index):
        if not hasattr(self, 'txn'):
            self.txn = open_lmdb(self.data_path)

        key = f'256-{str(index).zfill(5)}'.encode('utf-8')
        img_bytes = self.txn.get(key)

        buffer = BytesIO(img_bytes)
        image = Image.open(buffer)

        image = self.transform(image)
        gt = image.mul(0.5).add(0.5).mul(255).add(0.5).clamp(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()

        row = self.df.iloc[index]
        label = [0] * len(self.id_to_label)
        for k, v in row.items():
            label[self.label_to_id[k]] = int(v)

        return {
            "idx": index,
            "x_0": image,
            "gt": gt,
            "x_T": torch.randn(self.image_channel, self.image_size, self.image_size),
            "label": torch.tensor(label) # [-1, -1, 1, -1, 1, ...]
        }

    @staticmethod
    def collate_fn(batch):
        batch_size = len(batch)

        idx = []
        x_0 = []
        gts = []
        x_T = []
        label = []
        for i in range(batch_size):
            idx.append(batch[i]["idx"])
            x_0.append(batch[i]["x_0"])
            gts.append(batch[i]["gt"])
            x_T.append(batch[i]["x_T"])
            label.append(batch[i]["label"])

        x_0 = torch.stack(x_0, dim=0)
        x_T = torch.stack(x_T, dim=0)
        label = torch.stack(label, dim=0)

        return {
            "net_input": {
                "x_0": x_0,
                "x_T": x_T,
                "label": label,
            },
            "idx": idx,
            "gts": np.asarray(gts),
        }
