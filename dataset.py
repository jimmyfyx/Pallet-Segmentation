import torch
import os
import numpy as np

from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class WarehouseSegDataset(Dataset):
    def __init__(self, data_type, data_dir='/home/yixiaof2/projects/peer_robo/data_split_dino_v2/semseg'):
        # Input transformation defined by DINOv2
        self.transform_rgb = transforms.Compose([
            transforms.ToTensor(),
            lambda x: 255.0 * x[:3], # Discard alpha component and scale by 255
            transforms.Normalize(
                mean=(123.675, 116.28, 103.53),
                std=(58.395, 57.12, 57.375),
            ),
        ])

        self.rgb_dir = os.path.join(data_dir, data_type, 'rgb')
        self.ground_mask_dir = os.path.join(data_dir, data_type, 'ground_mask')
        self.pallet_mask_dir = os.path.join(data_dir, data_type, 'pallet_mask')
        self.data = []
        self.get_data_paths()
    
    def get_data_paths(self):
        for image_name in os.listdir(self.rgb_dir):
            rgb_path = os.path.join(self.rgb_dir, image_name)
            ground_mask_path = os.path.join(self.ground_mask_dir, image_name.replace('.jpg', '.npy'))
            pallet_mask_path = os.path.join(self.pallet_mask_dir, image_name.replace('.jpg', '.npy'))
            self.data.append((rgb_path, ground_mask_path, pallet_mask_path))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        rgb_path = self.data[idx][0]
        ground_mask_path = self.data[idx][1]
        pallet_mask_path = self.data[idx][2]
        rgb_image = np.array(Image.open(rgb_path).convert("RGB"))
        ground_mask = np.load(ground_mask_path)
        pallet_mask = np.load(pallet_mask_path)

        # Combine masks
        mask_tensor = torch.stack([
            torch.tensor(ground_mask, dtype=torch.float32),  # Channel 1: Ground
            torch.tensor(pallet_mask, dtype=torch.float32)   # Channel 2: Pallet
        ], dim=0)
        rgb_tensor = self.transform_rgb(rgb_image)

        return rgb_tensor, mask_tensor


