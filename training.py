import time
import random

import cv2
import numpy as np

import wandb

import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader, Subset

import torchvision.transforms as T
import torch.nn.functional as F

from tqdm import tqdm
from pathlib import Path

import h5py

from DepthAnythingV2.depth_anything_v2.dpt import DepthAnythingV2

# reference to cloned repository
from data.DeepFurniture.deepfurniture import DeepFurnitureDataset


DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'


SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.mps.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED) 


wandb.login()


class HDF5Dataset(Dataset):
    def __init__(self, h5_path):
        self.h5_path = h5_path
        with h5py.File(self.h5_path, 'r') as f:
            self.keys = list(f.keys())

    def __getitem__(self, index):
        if not hasattr(self, 'file') or self.file is None:
            self.file = h5py.File(self.h5_path, 'r')

        grp = self.file[self.keys[index]]
        image = torch.tensor(grp["image"][:])
        depth = torch.tensor(grp["depth"][:])
        return {"image": image, "depth": depth}

    def __len__(self):
        return len(self.keys)
    

def collate_fn(batch):
    return [item for item in batch if item is not None]


def preprocess_dataset_hdf5(dataset, split, image_size=(392, 392)):
    transform_image = T.Compose([
        T.Resize(image_size),
        T.ToTensor(),
        T.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])

    transform_depth = T.Compose([
        T.Resize(image_size),
        T.ToTensor()
    ])

    h5_path = f"processed_{split}_dataset_inverted_depths.h5"
    with h5py.File(h5_path, "w") as h5f:
        idx = 0
        for data in tqdm(dataset, desc=f"Preprocessing {split}"):
            try:
                if not data['image'] or not data['depth']:
                    continue

                image = transform_image(data['image'])
                depth = transform_depth(data['depth'])

                # Invert depth
                min_val = depth.min()
                max_val = depth.max()
                inverted_depth = max_val + min_val - depth

                # Save each sample as a group
                grp = h5f.create_group(f"sample_{idx}")
                grp.create_dataset("image", data=image.numpy(), compression="gzip")
                grp.create_dataset("depth", data=inverted_depth.numpy(), compression="gzip")

                idx += 1
            except Exception as e:
                print(f"Skipping corrupted sample: {e}")
                continue

    print(f"Saved HDF5 to: {h5_path}")



RUN_PREPROCESSING = False

if RUN_PREPROCESSING:
    base_dataset = DeepFurnitureDataset("./data/DeepFurniture/uncompressed_data")

    # Split indices
    num_samples = len(base_dataset)
    indices = list(range(num_samples))
    random.shuffle(indices)

    train_split = int(0.8 * num_samples)
    val_split = int(0.9 * num_samples)

    train_indices = indices[:train_split]
    val_indices = indices[train_split:val_split]
    test_indices = indices[val_split:]

    # Create subsets
    train_subset = Subset(base_dataset, train_indices)
    val_subset = Subset(base_dataset, val_indices)
    test_subset = Subset(base_dataset, test_indices)

    # Wrap subsets with image/depth transformations
    preprocess_dataset_hdf5(train_subset, 'train')
    preprocess_dataset_hdf5(val_subset, 'val')
    preprocess_dataset_hdf5(test_subset, 'test')



train_dataset = HDF5Dataset("data/DeepFurniture/pytorch/processed_train_dataset_inverted_depths.h5")
val_dataset = HDF5Dataset("data/DeepFurniture/pytorch/processed_val_dataset_inverted_depths.h5")
test_dataset = HDF5Dataset("data/DeepFurniture/pytorch/processed_test_dataset_inverted_depths.h5")


model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

encoder = 'vitl' # or 'vits', 'vitb', 'vitg'

model = DepthAnythingV2(**model_configs[encoder])
model.load_state_dict(torch.load(f'DepthAnythingV2/checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu'))
model = model.to(DEVICE).eval()



def compute_absrel(pred, gt, eps=1e-6):
    valid = gt > eps
    return (torch.abs(pred[valid] - gt[valid]) / gt[valid]).mean().item()



num_epochs = 20
run_name = f'depthanything-{encoder}-{time.strftime("%Y%m%d-%H%M%S")}'

train_loader = DataLoader(
    train_dataset,
    batch_size=2,
    shuffle=True,
    collate_fn=collate_fn
)
val_loader = DataLoader(
    val_dataset,
    batch_size=2,
    shuffle=False,
    collate_fn=collate_fn
)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
model.to(DEVICE)

wandb.init(project="hslu-dspro2-beyond2d", name=run_name, config={
    "epochs": num_epochs,
    "batch_size": train_loader.batch_size,
    "lr": optimizer.param_groups[0]["lr"],
    "encoder": encoder,
})

best_val_absrel = float("inf")

for epoch in range(num_epochs):
    model.train()
    epoch_loss = 0.0
    total_absrel_train = 0.0
    total_delta1_train = 0.0
    start_time = time.time()

    print(f"\n🚀 Starting Epoch {epoch+1}/{num_epochs}")
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False)

    for batch in progress_bar:
        images = torch.stack([b["image"] for b in batch]).to(DEVICE)
        depths = torch.stack([b["depth"].squeeze(0) for b in batch]).to(DEVICE)

        pred = model(images)
        loss = F.l1_loss(pred, depths)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

        absrel = compute_absrel(pred, depths)
        total_absrel_train += absrel

        ratio = torch.max(pred / depths.clamp(min=1e-6), depths / pred.clamp(min=1e-6))
        delta1 = (ratio < 1.25).float().mean().item()
        total_delta1_train += delta1

        progress_bar.set_postfix(loss=loss.item(), absRel=absrel, delta1=delta1)

    avg_loss = epoch_loss / len(train_loader)
    avg_absrel_train = total_absrel_train / len(train_loader)
    avg_delta1_train = total_delta1_train / len(train_loader)
    elapsed = time.time() - start_time
    print(f"✅ Epoch {epoch+1} | Loss: {avg_loss:.4f} | absRel: {avg_absrel_train:.4f} | delta1: {avg_delta1_train:.4f} | Time: {elapsed:.2f}s")

    # Validation
    model.eval()
    val_loss = 0.0
    total_absrel_val = 0.0
    total_delta1_val = 0.0

    with torch.no_grad():
        val_bar = tqdm(val_loader, desc="Validation", leave=False)
        for batch in val_bar:
            images = torch.stack([b["image"] for b in batch]).to(DEVICE)
            depths = torch.stack([b["depth"].squeeze(0) for b in batch]).to(DEVICE)

            pred = model(images)
            loss = F.l1_loss(pred, depths)
            val_loss += loss.item()

            absrel = compute_absrel(pred, depths)
            ratio = torch.max(pred / depths.clamp(min=1e-6), depths / pred.clamp(min=1e-6))
            delta1 = (ratio < 1.25).float().mean().item()

            total_absrel_val += absrel
            total_delta1_val += delta1

            val_bar.set_postfix(loss=loss.item(), absRel=absrel, delta1=delta1)

    avg_val_loss = val_loss / len(val_loader)
    avg_absrel_val = total_absrel_val / len(val_loader)
    avg_delta1_val = total_delta1_val / len(val_loader)
    print(f"🔍 Validation | Loss: {avg_val_loss:.4f} | absRel: {avg_absrel_val:.4f} | delta1: {avg_delta1_val:.4f}")

    wandb.log({
        "train_loss": avg_loss,
        "train_absRel": avg_absrel_train,
        "train_delta1": avg_delta1_train,
        "val_loss": avg_val_loss,
        "val_absRel": avg_absrel_val,
        "val_delta1": avg_delta1_val,
        "epoch": epoch + 1
    })

    if avg_absrel_val < best_val_absrel:
        best_val_absrel = avg_absrel_val
        
    checkpoint_path = Path(f"model_epoch_{epoch}.pth")
    torch.save(model.state_dict(), checkpoint_path)
    print(f"💾 Model saved with absRel={avg_absrel_val:.4f}")


wandb.finish()
