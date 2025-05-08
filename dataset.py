import os
from PIL import Image
import pandas as pd
from torch.utils.data import Dataset
from torchvision import transforms

class ArtDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.authors = sorted(self.data['artist'].unique())
        self.styles = sorted(self.data['style'].unique())
        self.author_to_idx = {author: idx for idx, author in enumerate(self.authors)}
        self.style_to_idx = {style: idx for idx, style in enumerate(self.styles)}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, os.path.basename(self.data.iloc[idx]['path']))
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        author = self.author_to_idx[self.data.iloc[idx]['artist']]
        style = self.style_to_idx[self.data.iloc[idx]['style']]
        return image, author, style
