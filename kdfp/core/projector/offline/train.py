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

from kdfp.schemas.subConfigs import TrainingParamsConfig
from kdfp.utils.data import ProjectorDataset
from .projectorDataModule import ProjectorDataModule
from .offlineProjector import OffProjModuleConfig, OfflineProjectorModule

from pathlib import Path
from typing import Union

from lightning.pytorch.callbacks import Callback
import ray.train
import uuid
import tempfile
import torch

def train_projector_per_worker(
        config: Dict[str, Any], 
        teacher_cache_dir: Union[str, Path],
        student_cache_dir: Union[str, Path],
        teacher_dim: int,
        student_dim: int,
        alignment_beta: float,
        training_params: TrainingParamsConfig,
        max_epochs: int,
        val_split: float = 0.1,
        test_split: float = 0.1,):
    """Training function for Ray Tune"""
    if training_params.reproducible_backend:
        ray.train.torch.enable_reproducibility()
    
    datamodule = ProjectorDataModule(
        teacher_cache_dir=teacher_cache_dir,
        student_cache_dir=student_cache_dir,
        batch_size=config["batch_size"],
        seed=training_params.seed,
        num_workers=training_params.resources.projector.num_data_workers,
        val_split=val_split,
        test_split=test_split,
    )
    cfg = OffProjModuleConfig(
        teacher_dim=teacher_dim,
        student_dim=student_dim,
        alignment_beta=alignment_beta,
        tied_weights=config["tied_weights"],
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
        torch_dtype=training_params.torch_dtype,
    )
    model = OfflineProjectorModule(train_config=cfg)
    
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
        deterministic=training_params.reproducible_backend,
        check_val_every_n_epoch=1,
    )
    trainer = prepare_trainer(trainer)
    
    trainer.fit(model, datamodule=datamodule)
    

