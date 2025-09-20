import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
# from kdfp.core.projector.offline import create_with_raytune
from pathlib import Path
import json
import wandb
from typing import Optional, Dict, Any, List, Literal
import pytorch_lightning as pl
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from kdfp.utils.data import ProjectorDataset
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from ray.tune import RunConfig
from ray.train import ScalingConfig, CheckpointConfig
from ray.train.torch import TorchTrainer
from functools import partial
from .projectorDataModule import ProjectorDataModule


OFF_PROJ_STATE_DICT_FILE = "offline_projector.bin"

class OfflineProjector(LightningModule):
    def __init__(self,
                 teacher_dim: int,
                 student_dim: int,
                 alignment_beta: float,
                #  hparams: Namespace,
                 lr: float,
                 weight_decay: float,
                 tied_weights: bool = False):
        super().__init__()
        # self.create = classmethod(create_with_raytune)
        self.save_hyperparameters()  #hparams)
        self.lr = lr
        self.weight_decay = weight_decay
        
        self.teacher_dim = teacher_dim
        self.student_dim = student_dim
        self.tied_weights = tied_weights
        
        # Encoder: teacher -> student space (i.e. DOWN)
        self.encoder = nn.Linear(teacher_dim, student_dim)
        
        # Decoder: student -> teacher space (i.e. UP)
        self.decoder = nn.Linear(student_dim, teacher_dim)
        
        # Weight tying if enabled
        if tied_weights:
            self.decoder.weight = nn.Parameter(self.encoder.weight.t())
        
        # Default loss weights
        self.alignment_beta = alignment_beta
        
        self.criterion = nn.MSELoss()

        self.val_total_losses = []
        self.test_total_losses = []


    def forward(self, x, direction: Optional[str] = None):
        """
        Forward pass with direction control
        
        Args:
            x: input tensor
            direction: 
                - "autoencode": teacher -> student -> teacher reconstruction
                - "encode": teacher -> student projection only
                - "decode": student -> teacher projection only
                - "align": student -> teacher alignment (for inference)
        """
        if direction == None:
            # Full autoencoder: teacher -> student -> teacher
            latent = self.encoder(x)
            reconstructed = self.decoder(latent)
            return reconstructed, latent
        
        elif direction == "down":
            # Teacher -> student projection
            return self.encoder(x)
        
        elif direction == "up":
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
            (1.0 - self.alignment_beta) * recon_loss +
            self.alignment_beta * align_loss
        )
        
        return {
            'total_loss': total_loss,
            'recon_loss': recon_loss,
            'align_loss': align_loss,
        }

    def training_step(self, batch, batch_idx):
        X_teacher, X_student = batch
        losses = self.compute_losses(X_teacher, X_student)
        
        # Log all losses
        self.log('train_total_loss', losses['total_loss'], prog_bar=True)
        self.log('train_recon_loss', losses['recon_loss'])
        self.log('train_align_loss', losses['align_loss'])
        
        return losses['total_loss']

    def validation_step(self, batch, batch_idx):
        X_teacher, X_student = batch
        losses = self.compute_losses(X_teacher, X_student)
        self.val_total_losses.append(losses['total_loss'])
        
        self.log('val_total_loss', losses['total_loss'], prog_bar=True)
        self.log('val_recon_loss', losses['recon_loss'])
        self.log('val_align_loss', losses['align_loss'])
        
        return losses['total_loss']

    def test_step(self, batch, batch_idx):
        X_teacher, X_student = batch
        
        # Compute all losses
        losses = self.compute_losses(X_teacher, X_student)
        self.test_total_losses.append(losses['total_loss'])
        
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
        self.log("kdfp_off_proj/val_total_loss", avg_val_total_loss, sync_dist=True)
        self.val_total_losses.clear()


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.lr, 
            weight_decay=self.weight_decay
        )
        # scheduler = 
        return optimizer

    def save_pretrained(self, save_path: str):
        """Save model with configuration"""
        # Create directory if it doesn't exist
        Path(save_path).mkdir(parents=True, exist_ok=True)
        
        # Save model state dict
        torch.save(self.state_dict(), f"{save_path}/{OFF_PROJ_STATE_DICT_FILE}")
        
        # Save configuration
        config = {
            'teacher_dim': self.teacher_dim,
            'student_dim': self.student_dim,
            'tied_weights': self.tied_weights,
            'alignment_beta': self.alignment_beta,
            'lr': self.lr,
            'weight_decay': self.weight_decay,
            # 'hparams': vars(self.hparams) if isinstance(self.hparams, Namespace) else dict(self.hparams),
        }
        
        with open(f"{save_path}/config.json", "w") as f:
            json.dump(config, f, indent=4)
        
        print(f"Model saved to {save_path}")
    
    @classmethod
    def load_pretrained(cls, save_path: str, map_location=None):
        """Load model from saved configuration"""
        # Load configuration
        with open(f"{save_path}/config.json", "r") as f:
            config = json.load(f)
        
        # # Extract hparams and convert back to Namespace if needed
        # hparams_dict = config.pop('hparams')
        # hparams = Namespace(**hparams_dict)
        
        # # Create model instance
        # model = cls(hparams=hparams, **config)
        model = cls(**config)
        
        # Load weights
        model_path = f"{save_path}/{OFF_PROJ_STATE_DICT_FILE}"
        if map_location:
            state_dict = torch.load(model_path, map_location=map_location)
        else:
            state_dict = torch.load(model_path)
        
        model.load_state_dict(state_dict)
        model.eval()
        
        print(f"Model loaded from {save_path}")
        return model
    
    def extract_single_direction(self, direction: Literal['up', 'down']):
        return self.encoder if direction == "down" else self.decoder
    
    @classmethod
    def create(self,
               teacher_cache_dir: str,
               student_cache_dir: str,
               teacher_dim: int,
               student_dim: int,
               alignment_beta: float,
               max_epochs: int,
            #    weight_decay_choices: List[float],
            #    max_grad_norm_choices: List[float],
               wd_min: float,
               wd_max: float,
               mgn_min: float,
               mgn_max: float,
               batch_sizes: List[int],
               lr_min: float = 1e-5,
               lr_max: float = 1e-2,
               wandb_project_name: str = "kdfp_offline_projector",
               n_trials: int = 15,
               val_split: float = 0.1,
               test_split: float = 0.1,
               save_dir: Optional[str] = None,
               seed: int = 44,
               reproducible_backend: bool = True,
               num_gpus_per_trial: float = 1.0,
               num_cpus_per_trial: int = 8,
               verbose: bool = True,
               # wandb_project: Optional[str] = None,
               # wandb_entity: Optional[str] = None,
               # wandb_run_name: Optional[str] = 'projector',
               # wandb_log_config: bool = True,
            #    wandb_log_gradients: bool = False,
               wandb_log_weights: bool = False,
               early_stopping_patience: Optional[int] = 5,) -> 'OfflineProjector':  #Tuple[Any, Dict[str, Any], Any, Dict[str, float], Dict[str, float]]:
        """Create and train projector with multi-objective loss"""
        from .train import train_projector_per_worker

        pl.seed_everything(seed)
        wandb_run_name = f'projector_{alignment_beta}'
        
        # dataset = ProjectorDataset(teacher_cache_dir, student_cache_dir)
        
        if verbose:
            # print(f"Dataset size: {len(dataset)}")
            print(f"Teacher dim: {teacher_dim}, Student dim: {student_dim}")
            print(f"Max epochs: {max_epochs}, Early stopping patience: {early_stopping_patience}")
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
        
        # Expanded search space with all hyperparameters as choices
        search_space: Dict[str, Any] = {
            "batch_size": tune.choice(batch_sizes),
            "learning_rate": tune.loguniform(lr_min, lr_max),
            "weight_decay": tune.loguniform(wd_min, wd_max),
            "max_grad_norm": tune.uniform(mgn_min, mgn_max),
            "tied_weights": tune.choice([True, False]),

        }
        train_func = partial(train_projector_per_worker,
                             teacher_cache_dir=teacher_cache_dir,
                             student_cache_dir=student_cache_dir,
                            #  dataset=dataset,
                             teacher_dim=teacher_dim,
                             student_dim=student_dim,
                             alignment_beta=alignment_beta,
                             max_epochs=max_epochs,
                             val_split=val_split,
                             test_split=test_split,
                             seed=seed,
                             reproducible_backend=reproducible_backend,)
        
        
        # reporter = CLIReporter(
        #     parameter_columns=["batch_size", "learning_rate", "max_grad_norm", "tied_weights"],
        #     metric_columns=["val_total_loss", "training_iteration"]
        # )
        scaling_config = ScalingConfig(
            num_workers=1, use_gpu=True, resources_per_worker={
                "CPU": num_cpus_per_trial,
                "GPU": num_gpus_per_trial,}
        )
        run_config = RunConfig(
            checkpoint_config=CheckpointConfig(
                num_to_keep=2,
                checkpoint_score_attribute="kdfp_off_proj/val_total_loss",
                checkpoint_score_order="min",
            ),
        )
        ray_ptl_trainer = TorchTrainer(
            train_func,
            scaling_config=scaling_config,
            run_config=run_config,
        )

        def tune_asha(num_samples: int):
            scheduler = ASHAScheduler(
                max_t=max_epochs,
                grace_period=1,
                reduction_factor=2
            )
            tuner = tune.Tuner(
                ray_ptl_trainer,
                param_space={"train_loop_config": search_space},
                tune_config=tune.TuneConfig(
                    metric="kdfp_off_proj/val_total_loss",
                    mode="min",
                    num_samples=num_samples,
                    scheduler=scheduler,
                ),
            )
            return tuner.fit()
        results = tune_asha(num_samples=n_trials)
        #---------------------------------------------------------------------------
        
        # analysis = tune.run(
        #     lambda config: train_projector_per_worker(
        #         config=config,
        #         dataset=dataset,
        #         teacher_dim=teacher_dim,
        #         student_dim=student_dim,
        #         max_epochs=max_epochs, 
        #         val_split=val_split,
        #         test_split=test_split,
        #         reproducible_backend=reproducible_backend,
        #     ),
        #     resources_per_trial={
        #         "cpu": num_cpus_per_trial,
        #         "gpu": num_gpus_per_trial
        #     },
        #     metric="val_total_loss",
        #     mode="min",
        #     config=search_space,
        #     num_samples=n_trials,
        #     scheduler=scheduler,
        #     progress_reporter=reporter,
        #     verbose=verbose,
        #     local_dir=save_dir,
        #     name=f"projector_{alignment_beta}_tuning"
        # )
        
        # # Get best configuration
        # best_train_config = analysis.best_config
        # best_val_loss = analysis.best_result["val_loss"]
        
        # if verbose:
        #     print(f"Best trial config: {best_train_config}")
        #     print(f"Best trial validation loss: {best_val_loss:.6f}")
        
        # # Prepare for final training
        # datamodule = ProjectorDataModule(
        #     dataset=dataset,
        #     batch_size=best_train_config["batch_size"],
        #     val_split=val_split,
        #     test_split=test_split,
        #     seed=seed
        # )
        # datamodule.setup()
        
        # # Train final model with wandb logging
        # if verbose:
        #     print("Training final model with best parameters...")
        
        # # Initialize wandb
        # wandb_config = {
        #     "teacher_dim": teacher_dim,
        #     "student_dim": student_dim,
        #     "alignment_beta": alignment_beta,
        #     "dataset_size": len(dataset),
        #     "max_epochs": max_epochs,
        #     "val_split": val_split,
        #     "test_split": test_split,
        #     "early_stopping_patience": early_stopping_patience,
        #     **best_train_config
        # }
        
        # wandb_logger = WandbLogger(
        #     project=wandb_project_name,
        #     # entity=wandb_entity,
        #     name=wandb_run_name,
        #     config=wandb_config,
        #     log_model="all" if wandb_log_weights else False,
        # )
        
        # best_model = OfflineProjector(
        #     teacher_dim=teacher_dim,
        #     student_dim=student_dim,
        #     alignment_beta=alignment_beta,
        #     tied_weights=best_train_config["tied_weights"],
        #     lr=best_train_config["learning_rate"],
        #     weight_decay=best_train_config["weight_decay"],
        #     max_grad_norm=best_train_config["max_grad_norm"]
        # )
        
        # # Create callbacks for validation and checkpointing
        # callbacks = []
        
        # # Model checkpoint callback to save best model based on validation loss
        # checkpoint_callback = ModelCheckpoint(
        #     monitor='val_total_loss',
        #     dirpath=f"{save_dir}/checkpoints",
        #     filename='best_model-{epoch:02d}-{val_total_loss:.4f}',
        #     save_top_k=1,
        #     mode='min',
        #     save_last=True,
        # )
        # callbacks.append(checkpoint_callback)
        
        # # Early stopping callback if specified
        # if early_stopping_patience:
        #     early_stop_callback = EarlyStopping(
        #         monitor='val_total_loss',
        #         patience=early_stopping_patience,
        #         verbose=verbose,
        #         mode='min'
        #     )
        #     callbacks.append(early_stop_callback)
        
        # final_trainer = pl.Trainer(
        #     max_epochs=max_epochs,  # Use the fixed max_epochs value
        #     accelerator="auto",
        #     devices="auto",
        #     enable_progress_bar=verbose,
        #     deterministic=True,
        #     check_val_every_n_epoch=1,
        #     logger=wandb_logger,
        #     gradient_clip_val=best_train_config["max_grad_norm"],
        #     callbacks=callbacks,
        #     log_every_n_steps=10 if verbose else 50,
        # )
        
        # if verbose:
        #     print(f"Starting final training with wandb logging...")
        #     print(f"W&B Run: {wandb_logger.experiment.name}")
        #     print(f"W&B Project: {wandb_logger.experiment.project}")
        #     if early_stopping_patience:
        #         print(f"Early stopping enabled with patience: {early_stopping_patience} epochs")
        
        # # Train with validation
        # final_trainer.fit(best_model, datamodule=datamodule)
        
        # # Load the best model from checkpoint
        # if checkpoint_callback.best_model_path:
        #     if verbose:
        #         print(f"Loading best model from: {checkpoint_callback.best_model_path}")
        #     best_model = OfflineProjector.load_from_checkpoint(
        #         checkpoint_callback.best_model_path,
        #         teacher_dim=teacher_dim,
        #         student_dim=student_dim,
        #         alignment_beta=alignment_beta,
        #     )
        
        # # Run validation on the best model
        # if verbose:
        #     print("Running final validation evaluation on best model...")
        
        # val_results = final_trainer.validate(best_model, datamodule=datamodule, verbose=verbose)
        # val_metrics = val_results[0] if val_results else {}
        
        # # Log validation metrics to wandb
        # if wandb_logger.experiment:
        #     wandb_logger.experiment.log({"val_metrics": val_metrics})
        #     for metric_name, metric_value in val_metrics.items():
        #         wandb_logger.experiment.log({f"final_val/{metric_name}": metric_value})
        
        # # Test evaluation
        # if verbose:
        #     print("Running final test evaluation on best model...")
        
        # test_results = final_trainer.test(best_model, datamodule=datamodule, verbose=verbose)
        # test_metrics = test_results[0] if test_results else {}
        
        # # Log test metrics to wandb
        # if wandb_logger.experiment:
        #     wandb_logger.experiment.log({"test_metrics": test_metrics})
        #     for metric_name, metric_value in test_metrics.items():
        #         wandb_logger.experiment.log({f"final_test/{metric_name}": metric_value})
        
        # # Compare validation and test results
        # comparison_metrics = {}
        # for metric_name in val_metrics.keys():
        #     if metric_name in test_metrics:
        #         val_value = val_metrics[metric_name]
        #         test_value = test_metrics[metric_name]
        #         comparison_metrics[f"val_{metric_name}"] = val_value
        #         comparison_metrics[f"test_{metric_name}"] = test_value
        #         comparison_metrics[f"diff_{metric_name}"] = abs(val_value - test_value)
        #         comparison_metrics[f"relative_diff_{metric_name}"] = abs(val_value - test_value) / max(abs(val_value), 1e-8)
        
        # # Log comparison metrics to wandb
        # if wandb_logger.experiment:
        #     wandb_logger.experiment.log({"comparison_metrics": comparison_metrics})
        #     wandb_logger.experiment.log({"val_test_comparison": comparison_metrics})
        
        # if verbose:
        #     print("\n" + "="*50)
        #     print("VALIDATION vs TEST RESULTS COMPARISON")
        #     print("="*50)
        #     for metric_name in val_metrics.keys():
        #         if metric_name in test_metrics:
        #             print(f"{metric_name}:")
        #             print(f"  Validation: {val_metrics[metric_name]:.6f}")
        #             print(f"  Test:       {test_metrics[metric_name]:.6f}")
        #             print(f"  Difference: {abs(val_metrics[metric_name] - test_metrics[metric_name]):.6f}")
        #             print()
        
        # # Save results
        # if save_dir:
        #     Path(save_dir).makedirs(parents=True, exist_ok=True)
            
        #     with open(f"{save_dir}/best_params.json", 'w') as f:
        #         json.dump(best_train_config, f, indent=4)
            
        #     with open(f"{save_dir}/val_metrics.json", 'w') as f:
        #         json.dump(val_metrics, f, indent=4)
            
        #     with open(f"{save_dir}/test_metrics.json", 'w') as f:
        #         json.dump(test_metrics, f, indent=4)
            
        #     with open(f"{save_dir}/comparison_metrics.json", 'w') as f:
        #         json.dump(comparison_metrics, f, indent=4)
            
        #     # torch.save(best_model.state_dict(), f"{save_dir}/state_dict.pt")
        #     best_model.save_pretrained(save_dir)
            
        #     # Save model to wandb if requested
        #     if wandb_log_weights and wandb_logger.experiment:
        #         model_artifact = wandb.Artifact(
        #             name=f"best_model_{wandb_logger.experiment.id}",
        #             type="model",
        #             description="Best model from hyperparameter tuning"
        #         )
        #         model_artifact.add_file(f"{save_dir}/{OFF_PROJ_STATE_DICT_FILE}")
        #         wandb_logger.experiment.log_artifact(model_artifact)
            
        #     if verbose:
        #         print(f"Results saved to {save_dir}/")
        
        # # Finish wandb run
        # wandb_logger.experiment.finish()
        
        # if verbose:
        #     print("Final training completed. W&B run finished.")
        
        # return best_model  #, best_train_config, analysis, val_metrics, test_metrics, comparison_metrics


# Add as class method
# Projector.create = classmethod(create_with_raytune)