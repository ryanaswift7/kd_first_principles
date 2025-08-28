import torch
from typing import Optional
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from kdfp.utils.data import ProjectorDataset


class ProjectorDataModule(pl.LightningDataModule):
    def __init__(self, dataset: ProjectorDataset, batch_size: int, seed: int,
                 val_split: float = 0.1, test_split: float = 0.1):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.val_split = val_split
        self.test_split = test_split
        self.seed = seed
        
    def setup(self, stage: Optional[str] = None):
        total_size = len(self.dataset)
        test_size = int(total_size * self.test_split)
        val_size = int(total_size * self.val_split)
        train_size = total_size - val_size - test_size
        
        generator = torch.Generator().manual_seed(self.seed)
        self.train_ds, self.val_ds, self.test_ds = random_split(
            self.dataset, 
            [train_size, val_size, test_size],
            generator=generator
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )