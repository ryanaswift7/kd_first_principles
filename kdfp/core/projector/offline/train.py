from typing import Dict, Any
import pytorch_lightning as pl
from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback
from kdfp.utils.data import ProjectorDataset
from .projectorDataModule import ProjectorDataModule
from .offlineProjector import OfflineProjector

def train_projector(config: Dict[str, Any], 
                   dataset: ProjectorDataset,
                   teacher_dim: int,
                   student_dim: int,
                   alignment_beta: float,
                   max_epochs: int = 10,
                   val_split: float = 0.1,
                   test_split: float = 0.1,
                   reproducible_backend: bool = True,):
    """Training function for Ray Tune"""

    
    datamodule = ProjectorDataModule(
        dataset=dataset,
        batch_size=config["batch_size"],
        val_split=val_split,
        test_split=test_split
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
    
    tune_callback = TuneReportCheckpointCallback(
        metrics={"val_total_loss": "val_total_loss"},
        filename="checkpoint",
        on="validation_end"
    )
    
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        devices="auto",
        callbacks=[tune_callback],
        enable_progress_bar=True,
        logger=False,
        enable_checkpointing=False,
        deterministic=reproducible_backend,
        check_val_every_n_epoch=1
    )
    
    trainer.fit(model, datamodule=datamodule)
    
    return trainer.callback_metrics["val_total_loss"].item()
