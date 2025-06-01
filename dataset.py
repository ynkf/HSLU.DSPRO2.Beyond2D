from torch.utils.data import Dataset

import torchvision.transforms as T

class DepthAnythingFurnitureDataset(Dataset):
    def __init__(self, dataset, image_size=(392, 392)):
        self.dataset = dataset
        self.transform_image = T.Compose([
            T.Resize(image_size),
            T.ToTensor(),
            T.Normalize(mean=[0.5]*3, std=[0.5]*3)
        ])
        self.transform_depth = T.Compose([
            T.Resize(image_size),
            T.ToTensor()
        ])
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        try:
            data = self.dataset[idx]
            if not data['depth'] or not data['image']:
                return None

            image = self.transform_image(data["image"])
            depth = self.transform_depth(data["depth"])
            return {"image": image, "depth": depth}
        except Exception as e:
            print(f"Skipping corrupted sample at index {idx}: {e}")
            return None