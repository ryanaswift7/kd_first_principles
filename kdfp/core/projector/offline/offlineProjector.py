import torch
import torch.nn as nn
from lightning.pytorch import LightningModule
# from kdfp.core.projector.offline import create_with_raytune
from pathlib import Path
import json
import os
import wandb
from typing import Optional, Dict, Any, List, Literal
import lightning.pytorch as pl
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from kdfp.utils.data import ProjectorDataset
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from ray.train import ScalingConfig, CheckpointConfig
from ray.train.torch import TorchTrainer, TorchConfig
from functools import partial

from kdfp.utils.enums import ProjectionDirection
from .projectorDataModule import ProjectorDataModule
import math
from ray.tune.integration.ray_train import TuneReportCallback
from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback
import ray.tune
import ray.train
import ray
from ray.train.lightning import RayTrainReportCallback
import uuid


OFF_PROJ_STATE_DICT_FILE = "offline_projector.bin"
AVAILABLE_GPUS = 2

class OfflineProjector(LightningModule):
    def __init__(self,
                 teacher_dim: int,
                 student_dim: int,
                 alignment_beta: float,
                #  hparams: Namespace,
                 lr: float,
                 weight_decay: float,
                 torch_dtype: torch.dtype,
                 tied_weights: bool = False):
        super().__init__()
        self.save_hyperparameters()
        
        # Encoder: teacher -> student space (i.e. DOWN)
        self.encoder = nn.Linear(teacher_dim, student_dim, dtype=torch_dtype)
        
        # Decoder: student -> teacher space (i.e. UP)
        self.decoder = nn.Linear(student_dim, teacher_dim, dtype=torch_dtype)
        
        # Weight tying if enabled
        if tied_weights:
            self.decoder.weight = nn.Parameter(self.encoder.weight.t())
        
        self.criterion = nn.MSELoss()

        self.val_total_losses = []
        self.test_total_losses = []


    def forward(self, x, direction: Optional[ProjectionDirection] = None):
        if direction == None:
            # Full autoencoder: teacher -> student -> teacher
            latent = self.encoder(x)
            reconstructed = self.decoder(latent)
            return reconstructed, latent
        
        elif direction == ProjectionDirection.DOWN:
            # Teacher -> student projection
            return self.encoder(x)
        
        elif direction == ProjectionDirection.UP:
            # Student -> teacher projection  
            return self.decoder(x)
        
        else:
            raise ValueError(f"Unknown direction: {direction}")

    def compute_losses(self, X_teacher, X_student):
        """Compute all loss components for multi-objective training"""
        
        # Autoencoder reconstruction loss
        reconstructed_teacher, latent_student = self(X_teacher, direction=None)
        recon_loss = self.criterion(reconstructed_teacher, X_teacher)
        
        # Representation alignment loss (student -> teacher space)
        align_loss = self.criterion(latent_student, X_student)
        
        # Total weighted loss
        total_loss = (
            (1.0 - self.hparams.alignment_beta) * recon_loss +
            self.hparams.alignment_beta * align_loss
        )
        
        return {
            'total_weighted_loss': total_loss,
            'raw_recon_loss': recon_loss,
            'raw_align_loss': align_loss,
        }

    def training_step(self, batch, batch_idx):
        X_teacher, X_student = batch
        losses = self.compute_losses(X_teacher, X_student)
        
        # Log all losses
        self.log('train_total_weighted_loss', losses['total_weighted_loss'], prog_bar=True, sync_dist=True)
        self.log('train_raw_recon_loss', losses['raw_recon_loss'], sync_dist=True)
        self.log('train_raw_align_loss', losses['raw_align_loss'], sync_dist=True)
        
        return losses['total_weighted_loss']

    def validation_step(self, batch, batch_idx):
        X_teacher, X_student = batch
        losses = self.compute_losses(X_teacher, X_student)
        self.val_total_losses.append(losses['total_weighted_loss'])
        
        self.log('val_total_weighted_loss', losses['total_weighted_loss'], prog_bar=True, sync_dist=True)
        self.log('val_raw_recon_loss', losses['raw_recon_loss'], sync_dist=True)
        self.log('val_raw_align_loss', losses['raw_align_loss'], sync_dist=True)
        
        return losses['total_weighted_loss']

    def test_step(self, batch, batch_idx):
        X_teacher, X_student = batch
        
        # Compute all losses
        losses = self.compute_losses(X_teacher, X_student)
        self.test_total_losses.append(losses['total_weighted_loss'])
        
        # Additional evaluation metrics
        with torch.no_grad():
            # Reconstruction quality
            reconstructed, latent = self(X_teacher, direction=None)
            recon_mae = nn.L1Loss()(reconstructed, X_teacher)
            recon_cosine = torch.nn.functional.cosine_similarity(reconstructed, X_teacher, dim=-1).mean()
            
            # Alignment quality
            align_l1 = nn.L1Loss()(latent, X_student)
            align_cosine = torch.nn.functional.cosine_similarity(latent, X_student, dim=-1).mean()
            
        # Log all metrics
        metrics = {
            **losses,
            'recon_mae': recon_mae,
            'recon_cosine': recon_cosine,
            'align_l1': align_l1,
            'align_cosine': align_cosine,
        }
        
        for name, value in metrics.items():
            self.log(f'test_{name}', value, sync_dist=True, on_epoch=True)
        
        return metrics
    
    def on_validation_epoch_end(self):
        avg_val_total_loss = torch.stack(self.val_total_losses).mean()
        self.log("kdfp_off_proj/val_total_weighted_loss", avg_val_total_loss, sync_dist=True)
        self.val_total_losses.clear()


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.hparams.lr, 
            weight_decay=self.hparams.weight_decay
        )
        # scheduler = 
        return optimizer

    def save_pretrained(self, path: str):
        """Save model with configuration"""
        # Create directory if it doesn't exist
        Path(path).mkdir(parents=True, exist_ok=True)
        
        # Save model state dict
        torch.save(self.state_dict(), f"{path}/{OFF_PROJ_STATE_DICT_FILE}")
        
        # Save configuration
        config = {
            'teacher_dim': self.hparams.teacher_dim,
            'student_dim': self.hparams.student_dim,
            'tied_weights': self.hparams.tied_weights,
            'alignment_beta': self.hparams.alignment_beta,
            'lr': self.hparams.lr,
            'weight_decay': self.hparams.weight_decay,
            'torch_dtype': str(self.hparams.torch_dtype).split('.')[-1],
            # 'hparams': vars(self.hparams) if isinstance(self.hparams, Namespace) else dict(self.hparams),
        }
        
        with open(f"{path}/config.json", "w") as f:
            json.dump(config, f, indent=4)
        
        print(f"Model saved to {path}")
    
    @classmethod
    def load_pretrained(cls, path: str, map_location=None):
        """Load model from saved configuration"""
        # Load configuration
        with open(f"{path}/config.json", "r") as f:
            config = json.load(f)
        
        # set dtype correctly
        config["torch_dtype"] = getattr(torch, config["torch_dtype"])
        
        # # Create model instance
        model = cls(**config)
        
        # Load weights
        model_path = f"{path}/{OFF_PROJ_STATE_DICT_FILE}"
        if map_location:
            state_dict = torch.load(model_path, map_location=map_location)
        else:
            state_dict = torch.load(model_path)
        
        model.load_state_dict(state_dict)
        model.eval()
        
        print(f"Model loaded from {path}")
        return model
    
    def extract_single_direction(self, direction: ProjectionDirection):
        return self.encoder if direction == ProjectionDirection.DOWN else self.decoder
    
    @classmethod
    def create(cls,
               teacher_cache_dir: str,
               student_cache_dir: str,
               teacher_dim: int,
               student_dim: int,
               torch_dtype: torch.dtype,
               alignment_beta: float,
               max_epochs: int,
               wd_min: float,
               wd_max: float,
               mgn_min: float,
               mgn_max: float,
               batch_sizes: List[int],
               lr_min: float = 1e-5,
               lr_max: float = 1e-2,
               n_trials: int = 15,
               val_split: float = 0.1,
               test_split: float = 0.1,
               seed: int = 44,
               reproducible_backend: bool = True,
               num_gpus_per_trial: float = 1.0,
               num_cpus_per_trial: int = 8,
               verbose: bool = True,) -> 'OfflineProjector':  #Tuple[Any, Dict[str, Any], Any, Dict[str, float], Dict[str, float]]:
        """Create and train projector with multi-objective loss"""
        from .train import train_projector_per_worker

        pl.seed_everything(seed, workers=True)
        run_name = f'off_proj_{alignment_beta}'
        
        # dataset = ProjectorDataset(teacher_cache_dir, student_cache_dir)
        
        if verbose:
            # print(f"Dataset size: {len(dataset)}")
            print(f"Teacher dim: {teacher_dim}, Student dim: {student_dim}")
            print(f"Max epochs: {max_epochs}")
            print(f"Validation split: {val_split}, Test split: {test_split}")
        
        # Validate input lists are not empty
        hparams = {
            'lr_min': lr_min,
            'lr_max': lr_max,
            'wd_min': wd_min,
            'wd_max': wd_max,
            'mgn_min': mgn_min,
            'mgn_max': mgn_max,
            'batch_sizes': batch_sizes,
        }
        if not all(hparams.values()):
            for k, v in hparams.items():
                if v is None:
                    print(f'{k} is empty')
                    raise ValueError("All hyperparameter options must be non-empty.")
        

        train_func = partial(train_projector_per_worker,
                             teacher_cache_dir=teacher_cache_dir,
                             student_cache_dir=student_cache_dir,
                             teacher_dim=teacher_dim,
                             student_dim=student_dim,
                             alignment_beta=alignment_beta,
                             max_epochs=max_epochs,
                             val_split=val_split,
                             test_split=test_split,
                             seed=seed,
                             reproducible_backend=reproducible_backend,
                             torch_dtype=torch_dtype,)
        
        # NOTE: CANNOT USE RAY TRAIN V2 (at least on single node)
        def tune_asha(num_trials: int):
            search_space: Dict[str, Any] = {
                "batch_size": tune.choice(batch_sizes),
                "learning_rate": tune.qloguniform(lr_min, lr_max, 5e-6),
                "weight_decay": tune.qloguniform(wd_min, wd_max, 5e-6),
                "max_grad_norm": tune.quniform(mgn_min, mgn_max, 1e-3),
                "tied_weights": tune.choice([True, False]),
            }
            scaling_config = ray.train.ScalingConfig(
                num_workers=1,
                use_gpu=True,
                resources_per_worker={
                    "CPU": num_cpus_per_trial,
                    "GPU": num_gpus_per_trial,},
                placement_strategy="PACK",
            )
            ray_ptl_trainer = TorchTrainer(
                train_func,
                scaling_config=scaling_config,
            )
            tune_run_config = ray.tune.RunConfig(
                name=f"{run_name}_tune_{uuid.uuid4().hex[:6]}",
                checkpoint_config=ray.tune.CheckpointConfig(
                    num_to_keep=1,
                    checkpoint_score_attribute="kdfp_off_proj/val_total_weighted_loss",
                    checkpoint_score_order="min",
                ),
                failure_config=ray.tune.FailureConfig(max_failures=5),
            )
            
            scheduler = ASHAScheduler(
                max_t=max_epochs,
                grace_period=1,
                reduction_factor=2
            )
            tuner = ray.tune.Tuner(
                ray_ptl_trainer,
                param_space={"train_loop_config": search_space},
                run_config=tune_run_config,
                tune_config=tune.TuneConfig(
                    metric="kdfp_off_proj/val_total_weighted_loss",
                    mode="min",
                    num_samples=num_trials,
                    scheduler=scheduler,
                    max_concurrent_trials=(AVAILABLE_GPUS // num_gpus_per_trial),
                ),
            )
            return tuner.fit()
        
        results = tune_asha(num_trials=n_trials)
        best_result = results.get_best_result()
        with best_result.checkpoint.as_directory() as checkpoint_dir:
            return OfflineProjector.load_from_checkpoint(f"{checkpoint_dir}/checkpoint.ckpt")
            