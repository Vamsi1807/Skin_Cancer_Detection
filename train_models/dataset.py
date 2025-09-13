import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import GroupShuffleSplit
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

class HAM10000Dataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row['image_id'] + '.jpg')
        image = Image.open(img_path).convert("RGB")
        image = np.array(image)
        label = int(row['label'])
        if self.transform:
            image = self.transform(image=image)['image']
        return image, label


def get_dataloaders(project_dir, batch_size=32, seed=42):
    csv_file = os.path.join(project_dir, "data", "ham10000", "HAM10000_metadata.csv")
    img_dir = os.path.join(project_dir, "data", "ham10000", "images")

    df = pd.read_csv(csv_file)

    if 'patient_id' not in df.columns:
        df['patient_id'] = df['image_id']

    label_map = {'nv':0,'mel':1,'bkl':2,'bcc':3,'akiec':4,'vasc':5,'df':6}
    df['label'] = df['dx'].map(label_map)

    gss = GroupShuffleSplit(n_splits=1, test_size=0.10, random_state=seed)
    train_idx, val_idx = next(gss.split(df, groups=df['patient_id']))
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df   = df.iloc[val_idx].reset_index(drop=True)

    # ✅ Lighter augmentation
    train_transform = A.Compose([
        A.RandomResizedCrop((224,224), scale=(0.85,1.0), p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.3),
        A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
        ToTensorV2(),
    ])
    
    val_transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
        ToTensorV2(),
    ])

    train_ds = HAM10000Dataset(train_df, img_dir, transform=train_transform)
    val_ds   = HAM10000Dataset(val_df, img_dir, transform=val_transform)

    # ✅ Weighted sampler (no class weights in loss)
    counts = train_df['label'].value_counts().sort_index().values
    class_weights = 1.0 / counts
    sample_weights = class_weights[train_df['label'].values]
    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                              num_workers=4, pin_memory=True)

    return train_loader, val_loader


