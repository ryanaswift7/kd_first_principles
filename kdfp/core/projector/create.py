import torch
import os
import json
import wandb
from typing import Optional, Dict, Any, Tuple, List
import pytorch_lightning as pl
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from kdfp.utils.data import ProjectorDataset
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from .train import train_projector
from .projectorDataModule import ProjectorDataModule
from .projector import Projector

def create_with_raytune(
    teacher_cache_dir: str,
    student_cache_dir: str,
    teacher_dim: int,
    student_dim: int,
    max_epochs: int,
    lr_choices: List[float],
    weight_decay_choices: List[float],
    max_grad_norm_choices: List[float],
    batch_size_choices: List[int],
    project_name: str = "bidirectional_projector",
    n_trials: int = 15,
    val_split: float = 0.1,
    test_split: float = 0.1,
    save_dir: Optional[str] = None,
    seed: int = 44,
    num_gpus_per_trial: float = 1.0,
    num_cpus_per_trial: int = 4,
    verbose: bool = True,
    wandb_project: Optional[str] = None,
    wandb_entity: Optional[str] = None,
    wandb_run_name: Optional[str] = None,
    wandb_log_config: bool = True,
    wandb_log_gradients: bool = False,
    wandb_log_weights: bool = False,
    early_stopping_patience: Optional[int] = 5,
) -> Tuple[Any, Dict[str, Any], Any, Dict[str, float], Dict[str, float]]:
    """Create and train bidirectional projector with multi-objective loss"""
    
    pl.seed_everything(seed)
    
    dataset = ProjectorDataset(teacher_cache_dir, student_cache_dir)
    
    if verbose:
        print(f"Dataset size: {len(dataset)}")
        print(f"Teacher dim: {teacher_dim}, Student dim: {student_dim}")
        print(f"Max epochs: {max_epochs}, Early stopping patience: {early_stopping_patience}")
        print(f"Validation split: {val_split}, Test split: {test_split}")
    
    # Validate input lists are not empty
    if not all([lr_choices, weight_decay_choices, max_grad_norm_choices, batch_size_choices]):
        raise ValueError("All hyperparameter choice lists must be non-empty")
    
    # Expanded search space with all hyperparameters as choices
    config: Dict[str, Any] = {
        "batch_size": tune.choice(batch_size_choices),
        "learning_rate": tune.choice(lr_choices),
        "weight_decay": tune.choice(weight_decay_choices),
        "max_grad_norm": tune.choice(max_grad_norm_choices),
        "tied_weights": tune.choice([True, False]),
        "recon_weight": tune.uniform(0.5, 2.0),
        "align_weight": tune.uniform(0.1, 1.0),
    }
    
    scheduler = ASHAScheduler(
        max_t=max_epochs,  # Use the fixed max_epochs value
        grace_period=2,
        reduction_factor=2
    )
    
    reporter = CLIReporter(
        parameter_columns=["batch_size", "learning_rate", "max_grad_norm", "tied_weights"],
        metric_columns=["val_total_loss", "training_iteration"]
    )
    
    analysis = tune.run(
        lambda config: train_projector(
            config=config,
            dataset=dataset,
            teacher_dim=teacher_dim,
            student_dim=student_dim,
            max_epochs=max_epochs,  # Use the fixed max_epochs value
            val_split=val_split,
            test_split=test_split
        ),
        resources_per_trial={
            "cpu": num_cpus_per_trial,
            "gpu": num_gpus_per_trial
        },
        metric="val_total_loss",
        mode="min",
        config=config,
        num_samples=n_trials,
        scheduler=scheduler,
        progress_reporter=reporter,
        verbose=verbose,
        local_dir=save_dir,
        name="bidirectional_projector_tuning"
    )
    
    # Get best configuration
    best_config = analysis.best_config
    best_val_loss = analysis.best_result["val_loss"]
    
    if verbose:
        print(f"Best trial config: {best_config}")
        print(f"Best trial validation loss: {best_val_loss:.6f}")
    
    # Prepare for final training
    datamodule = ProjectorDataModule(
        dataset=dataset,
        batch_size=best_config["batch_size"],
        val_split=val_split,
        test_split=test_split,
        seed=seed
    )
    datamodule.setup()
    
    # Train final model with wandb logging
    if verbose:
        print("Training final model with best parameters...")
    
    # Initialize wandb
    wandb_config = {
        "teacher_dim": teacher_dim,
        "student_dim": student_dim,
        "dataset_size": len(dataset),
        "max_epochs": max_epochs,
        "val_split": val_split,
        "test_split": test_split,
        "early_stopping_patience": early_stopping_patience,
        **best_config
    }
    
    wandb_logger = WandbLogger(
        project=wandb_project or project_name,
        entity=wandb_entity,
        name=wandb_run_name or f"best_model_final_training",
        config=wandb_config,
        log_model="all" if wandb_log_weights else False,
    )
    
    best_model = Projector(
        teacher_dim=teacher_dim,
        student_dim=student_dim,
        tied_weights=best_config["tied_weights"],
        lr=best_config["learning_rate"],
        weight_decay=best_config["weight_decay"],
        max_grad_norm=best_config["max_grad_norm"]
    )
    
    # Create callbacks for validation and checkpointing
    callbacks = []
    
    # Model checkpoint callback to save best model based on validation loss
    checkpoint_callback = ModelCheckpoint(
        monitor='val_total_loss',
        dirpath=save_dir if save_dir else './checkpoints',
        filename='best_model-{epoch:02d}-{val_total_loss:.4f}',
        save_top_k=1,
        mode='min',
        save_last=True,
    )
    callbacks.append(checkpoint_callback)
    
    # Early stopping callback if specified
    if early_stopping_patience:
        early_stop_callback = EarlyStopping(
            monitor='val_total_loss',
            patience=early_stopping_patience,
            verbose=verbose,
            mode='min'
        )
        callbacks.append(early_stop_callback)
    
    final_trainer = pl.Trainer(
        max_epochs=max_epochs,  # Use the fixed max_epochs value
        accelerator="auto",
        devices="auto",
        enable_progress_bar=verbose,
        deterministic=True,
        check_val_every_n_epoch=1,
        logger=wandb_logger,
        gradient_clip_val=best_config["max_grad_norm"],
        callbacks=callbacks,
        log_every_n_steps=10 if verbose else 50,
    )
    
    if verbose:
        print(f"Starting final training with wandb logging...")
        print(f"W&B Run: {wandb_logger.experiment.name}")
        print(f"W&B Project: {wandb_logger.experiment.project}")
        if early_stopping_patience:
            print(f"Early stopping enabled with patience: {early_stopping_patience} epochs")
    
    # Train with validation
    final_trainer.fit(best_model, datamodule=datamodule)
    
    # Load the best model from checkpoint
    if checkpoint_callback.best_model_path:
        if verbose:
            print(f"Loading best model from: {checkpoint_callback.best_model_path}")
        best_model = Projector.load_from_checkpoint(
            checkpoint_callback.best_model_path,
            teacher_dim=teacher_dim,
            student_dim=student_dim
        )
    
    # Run validation on the best model
    if verbose:
        print("Running final validation evaluation on best model...")
    
    val_results = final_trainer.validate(best_model, datamodule=datamodule, verbose=verbose)
    val_metrics = val_results[0] if val_results else {}
    
    # Log validation metrics to wandb
    if wandb_logger.experiment:
        wandb_logger.experiment.log({"val_metrics": val_metrics})
        for metric_name, metric_value in val_metrics.items():
            wandb_logger.experiment.log({f"final_val/{metric_name}": metric_value})
    
    # Test evaluation
    if verbose:
        print("Running final test evaluation on best model...")
    
    test_results = final_trainer.test(best_model, datamodule=datamodule, verbose=verbose)
    test_metrics = test_results[0] if test_results else {}
    
    # Log test metrics to wandb
    if wandb_logger.experiment:
        wandb_logger.experiment.log({"test_metrics": test_metrics})
        for metric_name, metric_value in test_metrics.items():
            wandb_logger.experiment.log({f"final_test/{metric_name}": metric_value})
    
    # Compare validation and test results
    comparison_metrics = {}
    for metric_name in val_metrics.keys():
        if metric_name in test_metrics:
            val_value = val_metrics[metric_name]
            test_value = test_metrics[metric_name]
            comparison_metrics[f"val_{metric_name}"] = val_value
            comparison_metrics[f"test_{metric_name}"] = test_value
            comparison_metrics[f"diff_{metric_name}"] = abs(val_value - test_value)
            comparison_metrics[f"relative_diff_{metric_name}"] = abs(val_value - test_value) / max(abs(val_value), 1e-8)
    
    # Log comparison metrics to wandb
    if wandb_logger.experiment:
        wandb_logger.experiment.log({"comparison_metrics": comparison_metrics})
        wandb_logger.experiment.log({"val_test_comparison": comparison_metrics})
    
    if verbose:
        print("\n" + "="*50)
        print("VALIDATION vs TEST RESULTS COMPARISON")
        print("="*50)
        for metric_name in val_metrics.keys():
            if metric_name in test_metrics:
                print(f"{metric_name}:")
                print(f"  Validation: {val_metrics[metric_name]:.6f}")
                print(f"  Test:       {test_metrics[metric_name]:.6f}")
                print(f"  Difference: {abs(val_metrics[metric_name] - test_metrics[metric_name]):.6f}")
                print()
    
    # Save results
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        
        with open(f"{save_dir}/best_params.json", 'w') as f:
            json.dump(best_config, f, indent=4)
        
        with open(f"{save_dir}/val_metrics.json", 'w') as f:
            json.dump(val_metrics, f, indent=4)
        
        with open(f"{save_dir}/test_metrics.json", 'w') as f:
            json.dump(test_metrics, f, indent=4)
        
        with open(f"{save_dir}/comparison_metrics.json", 'w') as f:
            json.dump(comparison_metrics, f, indent=4)
        
        torch.save(best_model.state_dict(), f"{save_dir}/best_weights.pt")
        best_model.save_pretrained(f"{save_dir}/best_model")
        
        # Save model to wandb if requested
        if wandb_log_weights and wandb_logger.experiment:
            model_artifact = wandb.Artifact(
                name=f"best_model_{wandb_logger.experiment.id}",
                type="model",
                description="Best model from hyperparameter tuning"
            )
            model_artifact.add_file(f"{save_dir}/best_weights.pt")
            wandb_logger.experiment.log_artifact(model_artifact)
        
        if verbose:
            print(f"Results saved to {save_dir}/")
    
    # Finish wandb run
    wandb_logger.experiment.finish()
    
    if verbose:
        print("Final training completed. W&B run finished.")
    
    return best_model, best_config, analysis, val_metrics, test_metrics, comparison_metrics