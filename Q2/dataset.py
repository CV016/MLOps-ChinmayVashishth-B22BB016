import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split

class CityscapesDataset(Dataset):
    def __init__(self, image_paths, mask_paths):
        self.image_paths = image_paths
        self.mask_paths = mask_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.image_paths[idx])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (128, 96), interpolation=cv2.INTER_NEAREST)
        img = img.astype(np.float32) / 255.0

        mask = cv2.imread(self.mask_paths[idx])
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        mask = cv2.resize(mask, (128, 96), interpolation=cv2.INTER_NEAREST)
        mask = np.max(mask, axis=-1)

        img = torch.from_numpy(img).permute(2, 0, 1)
        mask = torch.from_numpy(mask).long()

        return img, mask

def get_dataloaders(data_dir, batch_size=16, seed=42):
    img_dir = os.path.join(data_dir, "CameraRGB")
    mask_dir = os.path.join(data_dir, "CameraMask")

    img_files = sorted(os.listdir(img_dir))
    mask_files = sorted(os.listdir(mask_dir))

    img_paths = [os.path.join(img_dir, f) for f in img_files]
    mask_paths = [os.path.join(mask_dir, f) for f in mask_files]

    dataset = CityscapesDataset(img_paths, mask_paths)

    torch.manual_seed(seed)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=1)

    return train_loader, val_loader