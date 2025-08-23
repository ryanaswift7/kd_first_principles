import torch
import torch.nn as nn
import os
import wandb
import optuna
import torch.optim as optim
from utils.data import CachedActivationsDataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Optional
import json

class Projector(nn.Module):
    def __init__(self, teacher_dim: int, student_dim: int, tied_weights: bool = False):
        super(Projector, self).__init__()
        self.teacher_dim = teacher_dim
        self.student_dim = student_dim
        self.down = nn.Linear(teacher_dim, student_dim)
        self.up = nn.Linear(student_dim, teacher_dim)
        self.tied_weights = tied_weights
        if tied_weights:
            self.up.weight = nn.Parameter(self.down.weight.t())

    def forward(self, x):
        return self.up(self.down(x))
    
    def project(self, x, projection_direction: str):
        if projection_direction == "up":
            return self.up(x)
        elif projection_direction == "down":
            return self.down(x)
        else:
            raise ValueError(f"{projection_direction} is an invalid projection direction. Must be either 'up' or 'down'.")

    def save_pretrained(self, save_path: str):
        """Save model weights and config to specified path"""
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        
        # Save model weights
        weights_path = save_path
        torch.save(self.state_dict(), weights_path)
        
        # Save config
        config_path = save_path + 'config.json'
        config = {
            'teacher_dim': self.teacher_dim,
            'student_dim': self.student_dim,
            'tied_weights': self.tied_weights
        }
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)

    @classmethod
    def from_pretrained(cls, pretrained_path: str):
        """Load model from pretrained weights and config"""
        config_path = pretrained_path + 'config.json'
        
        if not os.path.exists(pretrained_path):
            raise FileNotFoundError(f"Pre-trained weights file not found at {pretrained_path}")
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found at {config_path}")
        
        # Load config
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Create model with config parameters
        model = cls(
            teacher_dim=config['teacher_dim'],
            student_dim=config['student_dim'],
            tied_weights=config['tied_weights']
        )
        
        # Load weights
        state_dict = torch.load(pretrained_path, weights_only=True)
        model.load_state_dict(state_dict)
        
        # Verify tied weights if specified in config
        if config['tied_weights']:
            if not torch.allclose(model.up.weight, model.down.weight.T):
                print("Warning: Tied weights specified in config but weight matrices are not transposes")
        
        return model

    
    @classmethod
    def create(
        cls,
        train_dataset: CachedActivationsDataset,
        teacher_dim: int,
        student_dim: int,
        project_name: str = "llm_kd_interp_projector",
        n_trials: int = 15,
        max_epochs: int = 10,  
        # final_training: bool = False,
        # Hyperparameter ranges
        lr_min: float = 1e-4,
        lr_max: float = 1e-2,
        weight_decay_min: float = 1e-6,
        weight_decay_max: float = 1e-2,
        # Other params
        batch_sizes: list = [32, 64, 128, 256],
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        verbose: bool = True,
        save_dir: Optional[str] = None,
        seed: int = 44,
    ):
        """
        Class method to create and train a Projector with hyperparameter optimization.
        
        Args:
            train_data: Tensor of shape (n_samples, teacher_dim) for training
            teacher_dim: Dimension of input data
            student_dim: Fixed bottleneck dimension
            project_name: W&B project name
            n_trials: Number of Optuna trials to run
            max_epochs: Maximum number of epochs per trial
            lr_min: Minimum learning rate for search
            lr_max: Maximum learning rate for search
            weight_decay_min: Minimum weight decay for search
            weight_decay_max: Maximum weight decay for search
            batch_sizes: List of batch sizes to try
            device: Device to train on
            verbose: Whether to print training progress
        
        Returns:
            tuple: (best_model, best_params, study)
        """
        # Initialize W&B
        # wandb.init(project=project_name, config={
        #     "teacher_dim": teacher_dim,
        #     "student_dim": student_dim,
        #     "param_ranges": {
        #         "learning_rate": [lr_min, lr_max],
        #         "weight_decay": [weight_decay_min, weight_decay_max]
        #     }
        # })
        
        # Move data to device
        # train_dataset = CachedActivationsDataset(train_data_cache)
        # train_dataset = train_dataset.to(device)
        
        def objective(trial):
            # Suggest hyperparameters within specified ranges
            trial_params = {
                "batch_size": trial.suggest_categorical('batch_size', batch_sizes),
                "learning_rate": trial.suggest_float('learning_rate', lr_min, lr_max, log=True),
                "weight_decay": trial.suggest_float('weight_decay', weight_decay_min, weight_decay_max, log=True),
                "tied_weights": trial.suggest_categorical('tied_weights', [True, False])
            }
            
            # Create model with fixed student_dim
            model = cls(
                teacher_dim=teacher_dim,
                student_dim=student_dim,
                tied_weights=trial_params["tied_weights"]
            ).to(device)
            
            train_loader = DataLoader(
                train_dataset,
                batch_size=trial_params["batch_size"],
                shuffle=True,
                # pin_memory=True,
                num_workers=8,
            )
            
            # Optimizer with current hyperparameters
            optimizer = optim.Adam(
                model.parameters(),
                lr=trial_params["learning_rate"],
                weight_decay=trial_params["weight_decay"]
            )
            criterion = nn.MSELoss()
            
            # # W&B run for this trial
            # trial_wandb = wandb.init(
            #     project=project_name,
            #     reinit=True,
            #     config={
            #         **trial_params,
            #         "student_dim": student_dim
            #     },
            #     group="optuna_trials"
            # )
            # Training loop
            model.train()
            for epoch in range(1):
                pbar = tqdm(train_loader, desc=f"Trial {trial.number}, Epoch {epoch+1}/{max_epochs}", leave=False)
                epoch_loss = 0.0
                
                for step, batch in enumerate(pbar):
                    batch = batch.to(device)
                    optimizer.zero_grad()
                    outputs = model(batch)
                    loss = criterion(outputs, batch)
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item() * batch.size(0)
                
                epoch_loss /= len(train_dataset)
                
                # Log metrics
                # trial_wandb.log({
                #     "loss": epoch_loss,
                #     "step": step,
                #     "epoch": epoch,
                # })
                
                # Report to Optuna
                trial.report(epoch_loss, epoch)
                
                if trial.should_prune():
                    # trial_wandb.finish()
                    raise optuna.TrialPruned()
            
            # trial_wandb.finish()
            return epoch_loss
        
        # custom sampler
        sampler = optuna.samplers.TPESampler(
        n_startup_trials=20,
        multivariate=True,
        seed=seed,  # For reproducibility
        )
        # Optuna study with W&B callback
        study = optuna.create_study(
            direction='minimize',
            pruner=optuna.pruners.MedianPruner(n_startup_trials=20),
            sampler=sampler,
        )
        
        # wandb_callback = optuna.integration.WeightsAndBiasesCallback(
        #     metric_name="loss",
        #     wandb_kwargs={"project": project_name}
        # )
        
        study.optimize(
            objective,
            n_trials=n_trials,
            n_jobs=1,
            # callbacks=[wandb_callback]
        )
        
        # Train final model with best params
        best_params = study.best_params
        best_model = cls(
            teacher_dim=teacher_dim,
            student_dim=student_dim,
            tied_weights=best_params['tied_weights']
        ).to(device)
        
        # Final W&B run
        # wandb.init(
        #     project=project_name,
        #     config={
        #         **best_params,
        #         "student_dim": student_dim
        #     },
        #     group="final_training",
        #     name="best_model"
        # )
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=best_params['batch_size'],
            shuffle=True,
            # pin_memory=True,
            num_workers=4,
        )
        
        optimizer = optim.Adam(
            best_model.parameters(),
            lr=best_params['learning_rate'],
            weight_decay=best_params['weight_decay']
        )
        criterion = nn.MSELoss()
        
        # Final training
        best_model.train()
        for epoch in range(max_epochs-1):
            running_loss = 0.0
            pbar = tqdm(train_loader, desc=f"Final Training Epoch {epoch+1}", disable=not verbose)
            
            for step, batch in enumerate(pbar):
                batch = batch.to(device)
                optimizer.zero_grad()
                outputs = best_model(batch)
                loss = criterion(outputs, batch)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item() * batch.size(0)
                pbar.set_postfix({'loss': loss.item()})
            
            epoch_loss = running_loss / len(train_dataset)
            # wandb.log({"final_loss": epoch_loss, "epoch": epoch})
            
            if verbose:
                print(f"Final Training Epoch {epoch+1}: Loss: {epoch_loss:.6f}")
        
        # Save and finish
        # projector_dir = f"{save_dir}/projector"
        os.makedirs(save_dir, exist_ok=True)
        # Save params to JSON file
        with open(f"{save_dir}/best_params.json", 'w') as f:
            json.dump(best_params, f, indent=4)
        torch.save(best_model.state_dict(), f"{save_dir}/best_weights.pt")
        print(f"Best projector model weights saved to {save_dir}/best_weights.pt")
        # wandb.save("best_model.pth")
        # wandb.finish()
        
        return best_model, best_params, study