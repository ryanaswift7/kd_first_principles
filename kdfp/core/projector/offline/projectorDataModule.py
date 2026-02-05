from pathlib import Path
from typing import Optional, Union

import lightning.pytorch as pl
import torch
from torch.utils.data import DataLoader, random_split

from kdfp.utils.data import ProjectorDataset


class ProjectorDataModule(pl.LightningDataModule):
    def __init__(
        self,
        teacher_cache_dir: Union[str, Path],
        student_cache_dir: Union[str, Path],
        batch_size: int,
        seed: int,
        num_workers: int,
        val_split: float = 0.1,
        test_split: float = 0.1,
        pin_memory: bool = True,
        prefetch_factor: int = 4,  # Added optimization parameter
    ):
        super().__init__()
        self.teacher_cache_dir = teacher_cache_dir
        self.student_cache_dir = student_cache_dir
        self.batch_size = batch_size
        self.val_split = val_split
        self.test_split = test_split
        self.seed = seed
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        # Optimization: Only use prefetch_factor if we actually have workers
        self.prefetch_factor = prefetch_factor if num_workers > 0 else None
        # Optimization: Keep workers alive to preserve memory maps/cache
        self.persistent_workers = True if num_workers > 0 else False

        self.dataset = None
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

    def setup(self, stage: Optional[str] = None):
        # Prevent re-initialization if setup is called multiple times (e.g. fit -> test)
        if self.dataset is not None:
            return

        # Initialize the optimized dataset
        # (This triggers the one-time metadata load on the main process)
        self.dataset = ProjectorDataset(
            teacher_cache_dir=str(self.teacher_cache_dir),
            student_cache_dir=str(self.student_cache_dir),
        )

        total_size = len(self.dataset)
        print(
            f"======================= Total dataset size: {total_size} samples ======================="
        )
        test_size = int(total_size * self.test_split)
        val_size = int(total_size * self.val_split)
        train_size = total_size - val_size - test_size

        # Verify sizes match (handle rounding errors)
        if train_size + val_size + test_size != total_size:
            train_size = total_size - val_size - test_size

        generator = torch.Generator().manual_seed(self.seed)

        # random_split creates Subset objects.
        # When combined with our ProjectorDataset, this is efficient because
        # it just stores indices, not data copies.
        self.train_ds, self.val_ds, self.test_ds = random_split(
            self.dataset, [train_size, val_size, test_size], generator=generator
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            prefetch_factor=self.prefetch_factor,
        )
