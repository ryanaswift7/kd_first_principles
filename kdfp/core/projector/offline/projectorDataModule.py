import torch
from typing import Optional, Union
from pathlib import Path
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from kdfp.utils.data import ProjectorDataset


class ProjectorDataModule(pl.LightningDataModule):
    def __init__(self,
                #  dataset: ProjectorDataset,
                 teacher_cache_dir: Union[str, Path],
                 student_cache_dir: Union[str, Path],
                 batch_size: int,
                 seed: int,
                 val_split: float = 0.1,
                 test_split: float = 0.1,
                 num_workers: int = 4,
                 pin_memory: bool = True):
        super().__init__()
        # self.dataset = dataset
        self.teacher_cache_dir = teacher_cache_dir
        self.student_cache_dir = student_cache_dir
        self.batch_size = batch_size
        self.val_split = val_split
        self.test_split = test_split
        self.seed = seed
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        
    def setup(self, stage: Optional[str] = None):
        self.dataset = ProjectorDataset(teacher_cache_dir=self.teacher_cache_dir,
                                        student_cache_dir=self.student_cache_dir)
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
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )