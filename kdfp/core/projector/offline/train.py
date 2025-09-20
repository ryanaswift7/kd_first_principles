from typing import Dict, Any
import pytorch_lightning as pl
from ray.train.lightning import (
    # TuneReportCheckpointCallback,
    RayDDPStrategy,
    RayLightningEnvironment,
    RayTrainReportCallback,
    prepare_trainer,
)
from ray.train import RunConfig, ScalingConfig, CheckpointConfig
from ray.train.torch import TorchTrainer
from pytorch_lightning.loggers.csv_logs import CSVLogger

from kdfp.utils.data import ProjectorDataset
from .projectorDataModule import ProjectorDataModule
from .offlineProjector import OfflineProjector

from ray.tune.integration.ray_train import TuneReportCallback
from pathlib import Path
from typing import Union


def train_projector_per_worker(config: Dict[str, Any], 
                #    dataset: ProjectorDataset,
                   teacher_cache_dir: Union[str, Path],
                   student_cache_dir: Union[str, Path],
                   teacher_dim: int,
                   student_dim: int,
                   alignment_beta: float,
                   seed: int,
                   max_epochs: int = 10,
                   val_split: float = 0.1,
                   test_split: float = 0.1,
                   reproducible_backend: bool = True,):
    """Training function for Ray Tune"""

    
    datamodule = ProjectorDataModule(
        # dataset=dataset,
        teacher_cache_dir=teacher_cache_dir,
        student_cache_dir=student_cache_dir,
        batch_size=config["batch_size"],
        val_split=val_split,
        test_split=test_split,
        seed=seed,
    )
    datamodule.setup()
    
    # Optional: Tune loss weights if desired
    # loss_weights = {
    #     'reconstruction': config.get("recon_weight"),
    #     'alignment': config.get("align_weight"),
    # }
    
    model = OfflineProjector(
        teacher_dim=teacher_dim,
        student_dim=student_dim,
        alignment_beta=alignment_beta,
        tied_weights=config["tied_weights"],
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
        # loss_weights=loss_weights
    )
    
    # tune_callback = TuneReportCheckpointCallback(
    #     metrics={"val_total_loss": "val_total_loss"},
    #     filename="checkpoint",
    #     on="validation_end"
    # )
    
    trainer = pl.Trainer(
        devices="auto",
        strategy=RayDDPStrategy(),
        plugins=[RayLightningEnvironment()],
        callbacks=[RayTrainReportCallback()],
        max_epochs=max_epochs,
        accelerator="auto",
        log_every_n_steps=100,
        logger=CSVLogger("logs"),
        # enable_progress_bar=False,
        # logger=False,
        enable_checkpointing=False,
        deterministic=reproducible_backend,
        check_val_every_n_epoch=1
    )
    trainer = prepare_trainer(trainer)
    
    # Train model
    trainer.fit(model, datamodule=datamodule)

    # Evaluation on the test dataset
    trainer.test(model, datamodule=datamodule)
    
    # return trainer.callback_metrics["val_total_loss"].item()
