"""
TODO:
- Teacher class which contains a dict of task names -> ft model locations
    - select task name to load ft model for task
- have an individual training function, and also one for applying indiv to the
    full task list
- should have create(task=task), init() [make and use task list], save, load
- init should make sure every task variant is trained and/or available
- change LM class to have a dict of task names and model locations
- default to HF for everything and use TL when necessary
"""


import os
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor
import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray.tune.schedulers import ASHAScheduler
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    get_cosine_schedule_with_warmup,
    get_cosine_with_hard_restarts_schedule_with_warmup,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset, Dataset
from typing import Dict, List, Optional, Union, Literal
import numpy as np
import wandb
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
from torchmetrics.classification import Accuracy
from torchmetrics import F1Score

from kdfp.schemas.tasks import TaskSpec
from kdfp.utils.data import load_task_data

class Teacher(pl.LightningModule):
    def __init__(
        self,
        model_name: str,
        tokenizer_name: str,
        task: TaskSpec,
        dataset_name: str,
        seq_len: int,
        batch_size: int,
        learning_rate: float,
        weight_decay: float,
        warmup_steps: int,
        num_workers: int,
        # project_name: str,
        # experiment_name: str,
        # scheduler_type: Literal["cosine", "cosine_warmup", "cosine_restarts", "cosine_annealing"],
        # num_cycles: int,
        # t_mult: int,
        # eta_min: float,
        # tune_mode: bool = False,
        n_samples: Optional[int] = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.model_name = model_name
        self.tokenizer_name = tokenizer_name
        self.task = task
        self.dataset_name = dataset_name
        self.seq_len = seq_len
        self.n_samples = n_samples
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        # self.scheduler_type = scheduler_type
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.num_workers = num_workers
        # self.tune_mode = tune_mode
        self.wandb_project_name = f"kdfp_teacher"
        self.wandb_experiment_name = task.name

        # self.num_cycles = num_cycles
        # self.t_mult = t_mult
        # self.eta_min = eta_min
        
        # Initialize model and tokenizer for causal LM
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            return_dict=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # Add padding token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Data collator for language modeling
        self.data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # Causal LM, not masked LM
            return_tensors="pt"
        )
        
        # Eval metrics
        binary_tasks = ["boolq", "winogrande", "piqa"]
        multiple_choice_tasks = ["arc", "hellaswag", "logiqa", "sciq"]
        if self.task.name in binary_tasks:
            self.acc = Accuracy(task='binary')
            self.f1 = F1Score(task='binary')
        elif self.task.name in multiple_choice_tasks:
            self.acc = Accuracy(task='multiclass')
            self.f1 = F1Score(task='multiclass')
        
    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
    
    def training_step(self, batch, batch_idx):
        outputs = self.model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels']
        )
        loss = outputs.loss
        
        
        self.log('train_loss', loss, prog_bar=True, logger=True)
        
        # Log current learning rate
        if self.trainer and self.trainer.optimizers:
            current_lr = self.trainer.optimizers[0].param_groups[0]['lr']
            self.log('learning_rate', current_lr, logger=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        outputs = self(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels']
        )
        loss = outputs.loss

        acc = self.acc(outputs, batch['labels'])
        f1 = self.f1(outputs, batch['labels'])
        
        self.log('val_loss', loss, prog_bar=True, logger=True)
        self.log('val_acc', acc, prog_bar=True, logger=True)
        self.log('val_f1', f1, prog_bar=True, logger=True)
        
        return {'val_loss': loss, 'val_acc': acc, 'val_f1': f1}
    
    def test_step(self, batch, batch_idx):
        outputs = self(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels']
        )
        loss = outputs.loss

        acc = self.acc(outputs, batch['labels'])
        f1 = self.f1(outputs, batch['labels'])
        
        # self.log('test_loss', loss, prog_bar=True, logger=True)
        self.log('test_acc', acc, prog_bar=True, logger=True)
        self.log('test_f1', f1, prog_bar=True, logger=True)
        
        return {'test_acc': acc, 'test_f1': f1}
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        return optimizer
        
        # # Calculate total training steps
        # if self.trainer:
        #     total_steps = self.trainer.estimated_stepping_batches
        # else:
        #     # Default estimate if trainer not available yet
        #     total_steps = 1000
        
        # # Choose the appropriate cosine scheduler
        # if self.scheduler_type == "cosine_warmup":
        #     # Cosine with warmup (from transformers)
        #     scheduler = get_cosine_schedule_with_warmup(
        #         optimizer,
        #         num_warmup_steps=self.warmup_steps,
        #         num_training_steps=total_steps,
        #         num_cycles=self.num_cycles
        #     )
        #     return {
        #         'optimizer': optimizer,
        #         'lr_scheduler': {
        #             'scheduler': scheduler,
        #             'interval': 'step',
        #             'frequency': 1
        #         }
        #     }
        
        # elif self.scheduler_type == "cosine_restarts":
        #     # Cosine with hard restarts and warmup (from transformers)
        #     scheduler = get_cosine_with_hard_restarts_schedule_with_warmup(
        #         optimizer,
        #         num_warmup_steps=self.warmup_steps,
        #         num_training_steps=total_steps,
        #         num_cycles=self.num_cycles
        #     )
        #     return {
        #         'optimizer': optimizer,
        #         'lr_scheduler': {
        #             'scheduler': scheduler,
        #             'interval': 'step',
        #             'frequency': 1
        #         }
        #     }
        
        # elif self.scheduler_type == "cosine_annealing":
        #     # Standard cosine annealing (from torch)
        #     scheduler = CosineAnnealingLR(
        #         optimizer,
        #         T_max=total_steps,
        #         eta_min=self.eta_min
        #     )
        #     return {
        #         'optimizer': optimizer,
        #         'lr_scheduler': {
        #             'scheduler': scheduler,
        #             'interval': 'step',
        #             'frequency': 1
        #         }
        #     }
        
        # elif self.scheduler_type == "cosine_warm_restarts":
        #     # Cosine annealing with warm restarts (from torch)
        #     scheduler = CosineAnnealingWarmRestarts(
        #         optimizer,
        #         T_0=total_steps // 10,
        #         T_mult=self.t_mult,
        #         eta_min=self.eta_min
        #     )
        #     return {
        #         'optimizer': optimizer,
        #         'lr_scheduler': {
        #             'scheduler': scheduler,
        #             'interval': 'step',
        #             'frequency': 1
        #         }
        #     }
        
        # else:  # "cosine" or default
        #     # Simple cosine decay without warmup
        #     def cosine_decay(step):
        #         progress = step / total_steps
        #         cosine_decay = 0.5 * (1 + torch.cos(torch.pi * progress))
        #         return cosine_decay
            
        #     scheduler = torch.optim.lr_scheduler.LambdaLR(
        #         optimizer,
        #         lr_lambda=cosine_decay
        #     )
        #     return {
        #         'optimizer': optimizer,
        #         'lr_scheduler': {
        #             'scheduler': scheduler,
        #             'interval': 'step',
        #             'frequency': 1
        #         }
        #     }
    
    # def _tokenize_function(self, examples):
    #     # Tokenize text for causal language modeling
    #     tokenized = self.tokenizer(
    #         examples['text'],
    #         padding=False,  # We'll pad in the collator
    #         truncation=True,
    #         max_length=self.max_seq_length,
    #         return_attention_mask=True,
    #         return_special_tokens_mask=False
    #     )
        
    #     # For causal LM, labels are the same as input_ids
    #     tokenized['labels'] = tokenized['input_ids'].copy()
        
    #     return tokenized
    
    # def prepare_data(self):
    #     """Load and tokenize the dataset for causal LM"""
    #     dataset = load_dataset(self.dataset_name)
        
    #     # Tokenize the dataset
    #     tokenized_datasets = dataset.map(
    #         self._tokenize_function,
    #         batched=True,
    #         remove_columns=dataset['train'].column_names,
    #         desc="Tokenizing dataset"
    #     )
        
    #     return tokenized_datasets
    
    def train_dataloader(self):
        dataset = load_task_data(
            task=self.task,
            tokenizer=self.tokenizer,
            seq_len=self.seq_len,
            split='train',
            n_samples=self.n_samples)
        
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=self.data_collator,
            pin_memory=True
        )
    
    def val_dataloader(self):
        dataset = load_task_data(
                    task=self.task,
                    tokenizer=self.tokenizer,
                    seq_len=self.seq_len,
                    split='validation',
                    n_samples=self.n_samples)
         
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.data_collator,
            pin_memory=True
        )
    
    def test_dataloader(self):
        dataset = load_task_data(
            task=self.task,
            tokenizer=self.tokenizer,
            seq_len=self.seq_len,
            split='test',
            n_samples=self.n_samples)
        
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self.data_collator,
            pin_memory=True
        )
    
    def train_model(
        self,
        max_epochs: int,
        gpus: int,
        tune: bool = True,
        # wandb_api_key: Optional[str] = None
    ):
        """Main training method with optional hyperparameter tuning"""
        
        # if wandb_api_key:
        #     wandb.login(key=wandb_api_key)
        
        if tune:
            self._run_hyperparameter_tuning(max_epochs, gpus)
        else:
            self._run_standard_training(max_epochs, gpus)
    
    def _run_standard_training(self, max_epochs: int, gpus: int):
        """Standard training without hyperparameter tuning"""
        raise NotImplementedError("Implement loading a config (default should be from tuning sweep) and train on it.")
        
        # # Initialize wandb logger
        # wandb_logger = WandbLogger(
        #     project=self.wandb_project_name,
        #     name=self.wandb_experiment_name,
        #     log_model=True,
        #     config=self.hparams
        # )
        
        # # Callbacks
        # callbacks = [
        #     EarlyStopping(monitor='val_loss', patience=3, mode='min'),
        #     ModelCheckpoint(
        #         dirpath='checkpoints',
        #         filename=f'best-model-{{epoch:02d}}-{{val_loss:.2f}}',  #-{self.scheduler_type}
        #         save_top_k=1,
        #         monitor='val_loss',
        #         mode='min'
        #     ),
        #     LearningRateMonitor(logging_interval='step')
        # ]
        
        # # Trainer
        # trainer = pl.Trainer(
        #     max_epochs=max_epochs,
        #     gpus=gpus,
        #     callbacks=callbacks,
        #     logger=wandb_logger,
        #     deterministic=True,
        #     progress_bar_refresh_rate=1,
        #     log_every_n_steps=10,
        #     accumulate_grad_batches=2 if self.batch_size < 32 else 1  # Gradient accumulation for small batches
        # )
        
        # # Log model architecture
        # wandb_logger.watch(self.model, log='gradients', log_freq=100)
        
        # # Train
        # trainer.fit(self, self.train_dataloader(), self.val_dataloader())
        
        # # Test
        # trainer.test(self, self.test_dataloader())
        
        # # Finish wandb run
        # wandb.finish()
    
    def _run_hyperparameter_tuning(
        self,
        max_epochs: int,
        gpus: int,
        tune_config: Dict
    ):
        """Hyperparameter tuning with Ray Tune"""
        
        if not ray.is_initialized():
            ray.init()
        
        # Tune configuration for causal LM
        config = {
            "learning_rate": tune.loguniform(1e-5, 1e-3),
            "weight_decay": tune.uniform(0.0, 0.1),
            "batch_size": tune.choice([8, 16, 32]),
            "max_seq_length": tune.choice([256, 512, 1024]),
            "warmup_steps": tune.choice([100, 500, 1000, 2000]),
            # "scheduler_type": tune.choice(["cosine_warmup", "cosine_restarts"]),
            # "num_cycles": tune.choice([1, 2, 3]),
            **tune_config
        }
        
        scheduler = ASHAScheduler(
            max_t=max_epochs,
            grace_period=1,
            reduction_factor=2
        )
        
        reporter = CLIReporter(
            parameter_columns=["learning_rate", "weight_decay", "batch_size"],  #, "scheduler_type"],
            metric_columns=["val_loss", "val_perplexity", "training_iteration"]
        )
        
        def train_tune(config, num_epochs=max_epochs, num_gpus=gpus):
            wandb.init(
                project=f"{self.wandb_project_name}-tune",
                config=config,
                reinit=True
            )
            
            model = Teacher(
                model_name=self.model_name,
                tokenizer_name=self.tokenizer_name,
                dataset_name=self.dataset_name,
                seq_len=config["max_seq_length"],
                batch_size=config["batch_size"],
                learning_rate=config["learning_rate"],
                weight_decay=config["weight_decay"],
                warmup_steps=config["warmup_steps"],
                num_workers=self.num_workers,
                # project_name=self.wandb_project_name,
                # experiment_name=self.wandb_experiment_name,
                # scheduler_type=config["scheduler_type"],
                # num_cycles=config.get("num_cycles", 1),
                # t_mult=self.t_mult,
                # eta_min=self.eta_min,
                # tune_mode=True
            )
            
            trainer = pl.Trainer(
                max_epochs=num_epochs,
                gpus=num_gpus,
                progress_bar_refresh_rate=0,
                callbacks=[
                    TuneReportCallback(
                        {
                            "val_loss": "val_loss",
                            "val_acc": "val_acc",
                            "val_f1": "val_f1",
                        },
                        on="validation_end"
                    ),
                    LearningRateMonitor(logging_interval='step')
                ]
            )
            
            trainer.fit(
                model,
                model.train_dataloader(),
                model.val_dataloader()
            )
            
            wandb.finish()
        
        analysis = tune.run(
            tune.with_parameters(
                train_tune,
                num_epochs=max_epochs,
                num_gpus=gpus
            ),
            resources_per_trial={"cpu": 2, "gpu": gpus},
            config=config,
            num_samples=8,
            scheduler=scheduler,
            progress_reporter=reporter,
            name="teacher_clm_tune",
            local_dir="./ray_results",
        )
        
        print("Best hyperparameters found were: ", analysis.best_config)
        wandb.init(project=f"{self.wandb_project_name}-best-config")
        wandb.config.update(analysis.best_config)
        wandb.finish()
    

    def save_model(self, path: str):
        """Save the model and tokenizer"""
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        
        # if hasattr(self, 'logger') and self.logger:
        #     artifact = wandb.Artifact('trained-model', type='model')
        #     artifact.add_dir(path)
        #     self.logger.experiment.log_artifact(artifact)
    
    @classmethod
    def load_model(cls, path: str, **kwargs):
        """Load a saved model"""
        model = AutoModelForCausalLM.from_pretrained(path)
        tokenizer = AutoTokenizer.from_pretrained(path)
        
        teacher = cls(**kwargs)
        teacher.model = model
        teacher.tokenizer = tokenizer
        
        return teacher
