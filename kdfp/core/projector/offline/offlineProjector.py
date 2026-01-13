import json
import uuid
from functools import partial

# from kdfp.core.projector.offline import create_with_raytune
from pathlib import Path
from typing import Any, Dict, Optional, Union

import lightning.pytorch as pl
import ray
import ray.train
import ray.tune
import torch
import torch.nn as nn
from lightning.pytorch import LightningModule
from ray import tune
from ray.train.torch import TorchTrainer
from ray.tune.schedulers import ASHAScheduler

from kdfp.schemas.configBase import ConfigBase
from kdfp.schemas.hparams import HparamsConfig
from kdfp.schemas.mixins import DtypeSerializerMixin
from kdfp.schemas.projections import ProjectionInstance
from kdfp.utils.enums import ProjectionDirection


def gen_off_proj_weights_filename(alignment_beta: float) -> str:
    return f"offline_projector_align_{alignment_beta}.bin"


class OffProjModuleConfig(ConfigBase, DtypeSerializerMixin):
    teacher_dim: int
    student_dim: int
    alignment_beta: float
    lr: float
    weight_decay: float
    torch_dtype: Union[str, torch.dtype]
    tied_weights: bool = False


class OfflineProjectorModule(LightningModule):
    def __init__(self, train_config: OffProjModuleConfig):
        super().__init__()
        # If train_config is a dict (from loading a checkpoint),
        # parse it back into a TeacherTrainConfig object.
        if isinstance(train_config, dict):
            self.train_config = OffProjModuleConfig(**train_config)
        else:
            # Otherwise (on initial creation), use the object directly.
            self.train_config = train_config

        # self.train_config = train_config
        self.save_hyperparameters({"train_config": self.train_config.model_dump()})

        # Encoder: teacher -> student space (i.e. DOWN)
        self.encoder = nn.Linear(
            self.train_config.teacher_dim,
            self.train_config.student_dim,
            dtype=self.train_config.torch_dtype,
        )

        # Decoder: student -> teacher space (i.e. UP)
        self.decoder = nn.Linear(
            self.train_config.student_dim,
            self.train_config.teacher_dim,
            dtype=self.train_config.torch_dtype,
        )

        # Weight tying if enabled
        if self.train_config.tied_weights:
            self.decoder.weight = nn.Parameter(self.encoder.weight.t())

        self.criterion = nn.MSELoss()

        self.val_total_losses = []
        self.test_total_losses = []

    @property
    def model_file(self):
        return gen_off_proj_weights_filename(
            alignment_beta=self.train_config.alignment_beta
        )

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
            1.0 - self.train_config.alignment_beta
        ) * recon_loss + self.train_config.alignment_beta * align_loss

        return {
            "total_weighted_loss": total_loss,
            "raw_recon_loss": recon_loss,
            "raw_align_loss": align_loss,
        }

    def training_step(self, batch, batch_idx):
        X_teacher, X_student = batch
        losses = self.compute_losses(X_teacher, X_student)

        # Log all losses
        self.log(
            "train_total_weighted_loss",
            losses["total_weighted_loss"],
            prog_bar=True,
            sync_dist=True,
        )
        self.log("train_raw_recon_loss", losses["raw_recon_loss"], sync_dist=True)
        self.log("train_raw_align_loss", losses["raw_align_loss"], sync_dist=True)

        return losses["total_weighted_loss"]

    def validation_step(self, batch, batch_idx):
        X_teacher, X_student = batch
        losses = self.compute_losses(X_teacher, X_student)
        self.val_total_losses.append(losses["total_weighted_loss"])

        self.log(
            "val_total_weighted_loss",
            losses["total_weighted_loss"],
            prog_bar=True,
            sync_dist=True,
        )
        self.log("val_raw_recon_loss", losses["raw_recon_loss"], sync_dist=True)
        self.log("val_raw_align_loss", losses["raw_align_loss"], sync_dist=True)

        return losses["total_weighted_loss"]

    def test_step(self, batch, batch_idx):
        X_teacher, X_student = batch

        # Compute all losses
        losses = self.compute_losses(X_teacher, X_student)
        self.test_total_losses.append(losses["total_weighted_loss"])

        # Additional evaluation metrics
        with torch.no_grad():
            # Reconstruction quality
            reconstructed, latent = self(X_teacher, direction=None)
            recon_mae = nn.L1Loss()(reconstructed, X_teacher)
            recon_cosine = torch.nn.functional.cosine_similarity(
                reconstructed, X_teacher, dim=-1
            ).mean()

            # Alignment quality
            align_l1 = nn.L1Loss()(latent, X_student)
            align_cosine = torch.nn.functional.cosine_similarity(
                latent, X_student, dim=-1
            ).mean()

        # Log all metrics
        metrics = {
            **losses,
            "recon_mae": recon_mae,
            "recon_cosine": recon_cosine,
            "align_l1": align_l1,
            "align_cosine": align_cosine,
        }

        for name, value in metrics.items():
            self.log(f"test_{name}", value, sync_dist=True, on_epoch=True)

        return metrics

    def on_validation_epoch_end(self):
        avg_val_total_loss = torch.stack(self.val_total_losses).mean()
        self.log(
            "kdfp_off_proj/val_total_weighted_loss", avg_val_total_loss, sync_dist=True
        )
        self.val_total_losses.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.train_config.lr,
            weight_decay=self.train_config.weight_decay,
        )
        # scheduler =
        return optimizer

    def extract_single_direction(self, direction: ProjectionDirection):
        return self.encoder if direction == ProjectionDirection.DOWN else self.decoder

    @classmethod
    def create(
        cls,
        teacher_cache_dir: str,
        student_cache_dir: str,
        teacher_dim: int,
        student_dim: int,
        projection: ProjectionInstance,
        hyperparams: HparamsConfig,
        training_params: TrainingParamsConfig,
        dirs: ExpDirConfig,
        val_split: float = 0.1,
        test_split: float = 0.1,
    ) -> (
        "OfflineProjectorModule"
    ):  # Tuple[Any, Dict[str, Any], Any, Dict[str, float], Dict[str, float]]:
        """Create and train projector with multi-objective loss"""
        from .train import train_projector_per_worker

        pl.seed_everything(training_params.seed, workers=True)
        run_name = f"off_proj_{projection.offline_alignment_beta}"

        train_func = partial(
            train_projector_per_worker,
            run_name=run_name,
            teacher_cache_dir=teacher_cache_dir,
            student_cache_dir=student_cache_dir,
            teacher_dim=teacher_dim,
            student_dim=student_dim,
            alignment_beta=projection.offline_projector_alignment_beta,
            max_epochs=projection.offline_projector_training_args.num_epochs,
            val_split=val_split,
            test_split=test_split,
            training_params=training_params,
        )

        # NOTE: CANNOT USE RAY TRAIN V2 (at least on single node)
        def tune_asha(num_trials: int):
            search_space: Dict[str, Any] = {
                "batch_size": tune.choice(
                    [2048]
                ),  # tune.choice(hyperparams.batch_sizes),
                "learning_rate": tune.qloguniform(
                    hyperparams.lr.mini, hyperparams.lr.maxi, 5e-7
                ),
                "weight_decay": tune.qloguniform(
                    hyperparams.weight_decay.mini, hyperparams.weight_decay.maxi, 5e-6
                ),
                "max_grad_norm": tune.quniform(
                    hyperparams.max_grad_norm.mini, hyperparams.max_grad_norm.maxi, 1e-3
                ),
                "tied_weights": tune.choice([True, False]),
            }
            scaling_config = ray.train.ScalingConfig(
                num_workers=training_params.resources.projector.num_ray_workers,
                use_gpu=True,
                resources_per_worker={
                    "CPU": training_params.resources.projector.num_cpus_per_trial,
                    "GPU": training_params.resources.projector.num_gpus_per_trial,
                },
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
                failure_config=ray.tune.FailureConfig(
                    max_failures=training_params.resources.max_failures
                ),
                storage_path=dirs.ray,
            )

            scheduler = ASHAScheduler(
                max_t=projection.offline_projector_training_args.num_epochs,
                grace_period=1,
                reduction_factor=2,
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
                    max_concurrent_trials=(
                        training_params.resources.num_available_gpus
                        // training_params.resources.projector.num_gpus_per_trial
                    ),
                ),
            )
            return tuner.fit()

        results = tune_asha(
            num_trials=projection.offline_projector_training_args.n_trials
        )
        best_result = results.get_best_result()
        with best_result.checkpoint.as_directory() as checkpoint_dir:
            return OfflineProjectorModule.load_from_checkpoint(
                f"{checkpoint_dir}/checkpoint.ckpt"
            )

    def save_pretrained(self, path: str):
        """Save model with configuration"""
        Path(path).mkdir(parents=True, exist_ok=True)
        save_path = f"{path}/{self.model_file}"
        torch.save(self.state_dict(), save_path)

        assert self.train_config

        with open(f"{path}/pl_config.json", "w") as f:
            # cfg = self.train_config.model_dump()
            # json.dump({"train_config": cfg}, f, indent=4)
            json.dump(self.train_config.model_dump(), f, indent=4)

        print(f"Offline projector saved to {path}")

    @classmethod
    def load_pretrained(cls, path: str):
        """Load model from saved configuration"""
        # Load configuration
        with open(f"{path}/pl_config.json", "r") as f:
            hparams = json.load(f)

        cfg = OffProjModuleConfig(**hparams)
        model = cls(train_config=cfg)

        # Load weights
        model_path = f"{path}/{model.model_file}"
        state_dict = torch.load(model_path, weights_only=True)

        model.load_state_dict(state_dict)
        model.eval()
        # with alignment beta {model.train_config.projection.offline_projector_alignment_beta}
        print(f"Loaded pretrained offline projector from {path}")
        return model
