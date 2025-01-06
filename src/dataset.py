import pandas as pd
from PIL import Image
import os.path as osp
from torch.utils.data import Dataset

class TerminatorDataset(Dataset):
    def __init__(self, img_dir, labels_path, transform=None):
        assert osp.exists(img_dir), f"Path {img_dir} does not exist"
        assert osp.exists(labels_path), f"Path {labels_path} does not exist"
        self.transform = transform
        self.img_dir = img_dir
        self.labels = pd.read_csv(
            labels_path, 
            header=None,
            names=['class', 'filename']
        )
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        label = self.labels.iloc[index]
        im_path = osp.join(self.img_dir, label.filename)
        assert osp.exists(im_path), f"Image not found: {im_path}"
        image = Image.open(im_path)
        if self.transform:
            image = self.transform(image)
        return image, label['class']
    
