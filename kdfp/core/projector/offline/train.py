from typing import Dict, Any
from lightning.pytorch import Trainer
from ray.train.lightning import (
    RayDDPStrategy,
    RayLightningEnvironment,
    RayTrainReportCallback,
    prepare_trainer,
)
from ray.train import RunConfig, ScalingConfig, CheckpointConfig
from ray.train.torch import TorchTrainer
# from pytorch_lightning.loggers.csv_logs import CSVLogger

from kdfp.utils.data import ProjectorDataset
from .projectorDataModule import ProjectorDataModule
from .offlineProjector import OfflineProjector

from pathlib import Path
from typing import Union

from lightning.pytorch.callbacks import Callback
import ray.train
import uuid
import tempfile
import torch

# class ModifiedRayTrainReportCallback(Callback):
#     def on_validation_epoch_end(self, trainer, pl_module):
#         with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
#             if ray.train.get_context().get_world_rank() == 0:
#                 callback_metrics = trainer.callback_metrics
#                 ptl_checkpoint = trainer.save_checkpoint(temp_checkpoint_dir)  # filepath=f"/research-storage/lab/ryanswif/kd_first_principles/results/lightning/checkpoints/{trainer.my_uuid}")
#                 checkpoint = ray.train.Checkpoint.from_directory(temp_checkpoint_dir)
#                 ray.train.report(callback_metrics, checkpoint=checkpoint)
#             else:
#                 ray.train.report(callback_metrics, checkpoint=None)


def train_projector_per_worker(config: Dict[str, Any], 
                #    dataset: ProjectorDataset,
                   teacher_cache_dir: Union[str, Path],
                   student_cache_dir: Union[str, Path],
                   teacher_dim: int,
                   student_dim: int,
                   alignment_beta: float,
                   seed: int,
                   torch_dtype: torch.dtype,
                   max_epochs: int = 10,
                   val_split: float = 0.1,
                   test_split: float = 0.1,
                   reproducible_backend: bool = True,):
    """Training function for Ray Tune"""
    if reproducible_backend:
        ray.train.torch.enable_reproducibility()
    
    datamodule = ProjectorDataModule(
        # dataset=dataset,
        teacher_cache_dir=teacher_cache_dir,
        student_cache_dir=student_cache_dir,
        batch_size=config["batch_size"],
        val_split=val_split,
        test_split=test_split,
        seed=seed,
    )
    
    model = OfflineProjector(
        teacher_dim=teacher_dim,
        student_dim=student_dim,
        alignment_beta=alignment_beta,
        tied_weights=config["tied_weights"],
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
        torch_dtype=torch_dtype,
    )
    
    trainer = Trainer(
        devices="auto",
        strategy=RayDDPStrategy(),
        plugins=[RayLightningEnvironment()],
        callbacks=[RayTrainReportCallback()],
        max_epochs=max_epochs,
        accelerator="auto",
        gradient_clip_algorithm='norm',
        gradient_clip_val=config["max_grad_norm"],
        enable_progress_bar=False,
        enable_checkpointing=False,
        deterministic=reproducible_backend,
        check_val_every_n_epoch=1,
    )
    trainer = prepare_trainer(trainer)
    
    trainer.fit(model, datamodule=datamodule)
    

