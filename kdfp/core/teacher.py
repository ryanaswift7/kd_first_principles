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
from pydantic import field_validator
import torch
from torch.utils.data import DataLoader
import lightning.pytorch as pl
# import pytorch_lightning as pl
import ray
from ray.tune import CLIReporter
from ray.tune.integration.pytorch_lightning import TuneReportCallback, TuneReportCheckpointCallback
from ray.tune.schedulers import ASHAScheduler
import torchmetrics
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

from kdfp.schemas.configBase import ConfigBase
from kdfp.schemas.subConfigs import DtypeSerializerMixin, HyperparamsConfig, LMParamsConfig, TrainingParamsConfig
from kdfp.schemas.tasks import TaskSpec
from kdfp.utils.data import load_task_data
import json
from ray.train.lightning import (
    RayDDPStrategy,
    RayFSDPStrategy,
    RayLightningEnvironment,
    RayTrainReportCallback,
    prepare_trainer,
)
import ray.tune
import ray.train
import uuid

def gen_teacher_weights_file(task_name: str) -> str:
    return f"teacher_{task_name}.bin"

class TeacherTrainConfig(ConfigBase, DtypeSerializerMixin):
    model_name: str
    tokenizer_name: str
    task: TaskSpec
    seq_len: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    max_grad_norm: float
    seed: int
    torch_dtype: Union[str, torch.dtype]
    num_workers: Optional[int] = None
    n_samples: Optional[int] = None
    # warmup_steps: int,
    # project_name: str,
    # experiment_name: str,
    # scheduler_type: Literal["cosine", "cosine_warmup", "cosine_restarts", "cosine_annealing"],
    # num_cycles: int,
    # t_mult: int,
    # eta_min: float,
    # tune_mode: bool = False,
    @field_validator('torch_dtype')
    def validate_dtype(cls, v):
        if isinstance(v, str):
            v = getattr(torch, v)
        return v

class Teacher(pl.LightningModule):
    def __init__(self, train_config):
        super().__init__()

        # If train_config is a dict (from loading a checkpoint),
        # parse it back into a TeacherTrainConfig object.
        if isinstance(train_config, dict):
            self.train_config = TeacherTrainConfig(**train_config)
        else:
            # Otherwise (on initial creation), use the object directly.
            self.train_config = train_config

        # Now self.train_config is guaranteed to be a TeacherTrainConfig object,
        # and you can safely call .model_dump()
        self.save_hyperparameters({"train_config": self.train_config.model_dump()})

        
        # Initialize model and tokenizer for causal LM
        self.model = AutoModelForCausalLM.from_pretrained(
            self.train_config.model_name,
            dtype=self.train_config.torch_dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(self.train_config.tokenizer_name)
        
        # Add padding token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Data collator for language modeling
        self.data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # Causal LM, not masked LM
            return_tensors="pt"
        )
        
        if self.train_config.task.task_type == 'multiclass':
            acc = Accuracy(task=self.train_config.task.task_type, num_classes=4)
            f1 = F1Score(task=self.train_config.task.task_type, num_classes=4)
        else:  # binary
            acc = Accuracy(task=self.train_config.task.task_type)
            f1 = F1Score(task=self.train_config.task.task_type)

        # Eval metrics
        self.val_metrics = torchmetrics.MetricCollection(
            {
                "acc": acc,
                "f1": f1,
            },
            prefix="val_"
        )
        self.test_metrics = self.val_metrics.clone(prefix='test_')

        
    @property
    def model_file(self):
        return gen_teacher_weights_file(self.train_config.task.task_name)
        
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
        print("\n\nRAN TRAINING STEP\n\n")

        # if batch_idx == 0:
        #     # Get the first item's details from the batch
        #     first_input_ids = batch['input_ids'][0]
        #     # last_5_input_ids = first_input_ids[-5:]
            
        #     # Decode for readability
        #     context_text = self.tokenizer.decode(first_input_ids, skip_special_tokens=True)

        #     # Print in a clean, readable format
        #     print("\n" + "="*60)
        #     print("🔍 TRAIN EXAMPLE (Batch 0, Sample 0)")
        #     print(f"  INPUT CONTEXT:\n    '{context_text}'")
        #     print("="*60 + "\n")

        self.log('train_loss', loss, sync_dist=True)
        
        return loss
        
    def validation_step(self, batch, batch_idx):
        """
        Corrected, efficient, and vectorized validation step.
        """
        # The batch contains tensors of shape [batch_size, 4, seq_len]
        batch_size, num_choices, seq_len = batch['input_ids'].shape
        true_labels = batch['label']

        # 1. Reshape inputs for a single, efficient forward pass
        # Shape becomes [batch_size * 4, seq_len]
        input_ids = batch['input_ids'].view(-1, seq_len)
        # attention_mask = batch['attention_mask'].view(-1, seq_len)

        # 2. Perform one forward pass for all choices in the batch
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids)  # , attention_mask=attention_mask)
        
        logits = outputs.logits

        # --- Add the debugging check for safety ---
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            raise ValueError("NaN or Inf in logits, stopping validation.")

        # 3. Calculate the Negative Log-Likelihood (NLL) for each choice
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()

        # Use reduction='none' to get the loss for each token individually
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        
        nll_per_token = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        # Reshape the per-token losses to match the sequence structure
        nll_per_token = nll_per_token.view(batch_size * num_choices, -1)
        
        # Ignore the loss for padding tokens by setting them to zero
        nll_per_token[shift_labels == self.tokenizer.pad_token_id] = 0
        
        # Sum the NLL for each sequence to get a score per choice
        nll_scores = nll_per_token.sum(dim=1)

        # 4. Reshape scores and find the prediction
        # Reshape scores from [batch_size * 4] to [batch_size, 4]
        choice_nll_scores = nll_scores.view(batch_size, num_choices)

        # The prediction is the choice with the minimum NLL
        preds = torch.argmin(choice_nll_scores, dim=1)

        # 5. Update metrics and log loss
        self.val_metrics.update(preds, true_labels)
        
        # Log the mean NLL of the correct choices
        correct_choice_loss = choice_nll_scores[torch.arange(batch_size), true_labels].mean()
        self.log('val_loss', correct_choice_loss, sync_dist=True)

        return {"val_loss": correct_choice_loss}
    
    def on_validation_epoch_end(self):
        self.log_dict(self.val_metrics.compute(), sync_dist=True)
        self.val_metrics.reset()

    def test_step(self, batch, batch_idx):
        # The batch contains tensors of shape [batch_size, 4, seq_len]
        batch_size, num_choices, seq_len = batch['input_ids'].shape
        true_labels = batch['label']

        # 1. Reshape inputs for a single, efficient forward pass
        # Shape becomes [batch_size * 4, seq_len]
        input_ids = batch['input_ids'].view(-1, seq_len)
        # attention_mask = batch['attention_mask'].view(-1, seq_len)

        # 2. Perform one forward pass for all choices in the batch
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids)  #, attention_mask=attention_mask)
        
        logits = outputs.logits

        # --- Add the debugging check for safety ---
        if torch.isnan(logits).any() or torch.isinf(logits).any():
            raise ValueError("NaN or Inf in logits, stopping testing.")

        # 3. Calculate the Negative Log-Likelihood (NLL) for each choice
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()

        # Use reduction='none' to get the loss for each token individually
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        
        nll_per_token = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        # Reshape the per-token losses to match the sequence structure
        nll_per_token = nll_per_token.view(batch_size * num_choices, -1)
        
        # Ignore the loss for padding tokens by setting them to zero
        nll_per_token[shift_labels == self.tokenizer.pad_token_id] = 0
        
        # Sum the NLL for each sequence to get a score per choice
        nll_scores = nll_per_token.sum(dim=1)

        # 4. Reshape scores and find the prediction
        # Reshape scores from [batch_size * 4] to [batch_size, 4]
        choice_nll_scores = nll_scores.view(batch_size, num_choices)

        # The prediction is the choice with the minimum NLL
        preds = torch.argmin(choice_nll_scores, dim=1)

        # 5. Update metrics and log loss
        self.test_metrics.update(preds, true_labels)
        
        # Log the mean NLL of the correct choices
        correct_choice_loss = choice_nll_scores[torch.arange(batch_size), true_labels].mean()
        self.log('test_loss', correct_choice_loss, sync_dist=True)

        return {"test_loss": correct_choice_loss}

    def on_test_epoch_end(self):
        self.log_dict(self.test_metrics.compute(), sync_dist=True)
        self.test_metrics.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.train_config.learning_rate,
            weight_decay=self.train_config.weight_decay
        )
        return optimizer
        
    def train_dataloader(self):
        dataset = load_task_data(
            task=self.train_config.task,
            tokenizer=self.tokenizer,
            seq_len=self.train_config.seq_len,
            split='train',
            n_samples=self.train_config.n_samples)
        
        # dataset = dataset.shuffle(seed=self.train_config.seed, buffer_size=1000)
        
        return DataLoader(
            dataset,
            batch_size=self.train_config.batch_size,
            shuffle=False,
            num_workers=self.train_config.num_workers,
            collate_fn=self.data_collator,
            pin_memory=True
        )
    # NOTE: can't use self.data_collator bc we tokenized in a custom format
    def val_dataloader(self):
        dataset = load_task_data(
                    task=self.train_config.task,
                    tokenizer=self.tokenizer,
                    seq_len=self.train_config.seq_len,
                    split='validation',
                    n_samples=self.train_config.n_samples)

        return DataLoader(
            dataset,
            batch_size=self.train_config.batch_size,
            shuffle=False,
            num_workers=self.train_config.num_workers,
            # collate_fn=self.data_collator,
            pin_memory=True
        )
    
    def test_dataloader(self):
        dataset = load_task_data(
            task=self.train_config.task,
            tokenizer=self.tokenizer,
            seq_len=self.train_config.seq_len,
            split='test',
            n_samples=self.train_config.n_samples)
        
        return DataLoader(
            dataset,
            batch_size=self.train_config.batch_size,
            shuffle=False,
            num_workers=self.train_config.num_workers,
            # collate_fn=self.data_collator,
            pin_memory=True
        )
    @classmethod
    def create(
        cls,
        lm_name: str,
        task: TaskSpec,
        lm_params: LMParamsConfig,
        hyperparams: HyperparamsConfig,
        training_params: TrainingParamsConfig,
        torch_dtype: torch.dtype,
    ):
        """Create a teacher for a given task using hyperparameter tuning with Ray Tune"""
        # os.environ["RAY_TRAIN_V2_ENABLED"] = '1'
        # torch.set_float32_matmul_precision('medium')
        pl.seed_everything(training_params.seed, workers=True)
        run_name = f'teacher_{task.task_name}'
        
        def train_teacher_per_worker(config):
            if training_params.reproducible_backend:
                ray.train.torch.enable_reproducibility()
            train_config = TeacherTrainConfig(
                model_name=lm_name,
                tokenizer_name=lm_name,
                task=task,
                seq_len=training_params.seq_len,
                batch_size=config["batch_size"],
                learning_rate=config["learning_rate"],
                weight_decay=config["weight_decay"],
                max_grad_norm=config["max_grad_norm"],
                n_samples=lm_params.num_samples,
                num_workers=training_params.num_data_workers,
                seed=training_params.seed,
                torch_dtype=torch_dtype,
            )

            model = Teacher(train_config=train_config)
            
            trainer = pl.Trainer(
                devices="auto",
                accelerator="auto",
                strategy=RayDDPStrategy(),
                plugins=[RayLightningEnvironment()],
                callbacks=[RayTrainReportCallback()],
                max_epochs=lm_params.num_epochs,
                gradient_clip_algorithm='norm',
                gradient_clip_val=config["max_grad_norm"],
                enable_progress_bar=False,
                enable_checkpointing=False,
                deterministic=training_params.reproducible_backend,
                check_val_every_n_epoch=1,
            )
            trainer = prepare_trainer(trainer)
            model.train()
            trainer.fit(model)
            
        def tune_asha(num_trials: int):
            
            train_loop_config = {
                "learning_rate": ray.tune.qloguniform(hyperparams.lr.mini, hyperparams.lr.maxi, 1e-6),
                "weight_decay": ray.tune.quniform(hyperparams.weight_decay.mini, hyperparams.weight_decay.maxi, 1e-4),
                "batch_size": ray.tune.choice(hyperparams.batch_sizes),
                "max_grad_norm": ray.tune.qloguniform(hyperparams.max_grad_norm.mini, hyperparams.max_grad_norm.maxi, 1e-1)
            }
            scaling_config = ray.train.ScalingConfig(
                num_workers=1,
                use_gpu=True,
                resources_per_worker={
                    "CPU": lm_params.num_cpus_per_trial,
                    "GPU": lm_params.num_gpus_per_trial,},
                placement_strategy="PACK",
            )
            tune_run_config = ray.tune.RunConfig(
                name=f"{run_name}_tune_{uuid.uuid4().hex[:6]}",
                checkpoint_config=ray.tune.CheckpointConfig(
                    num_to_keep=1,
                    checkpoint_frequency=0,
                    checkpoint_score_attribute="val_acc",
                    checkpoint_score_order="max",
                ),
                failure_config=ray.tune.FailureConfig(max_failures=5),
                storage_path="/research-storage/lab/ryanswif/kd_first_principles/ray-shenan/"
            )
            ray_ptl_trainer = ray.train.torch.TorchTrainer(
                train_teacher_per_worker,
                scaling_config=scaling_config,
            )
            
            scheduler = ASHAScheduler(
                max_t=lm_params.num_epochs,
                grace_period=1,
                reduction_factor=2
            )
            tuner = ray.tune.Tuner(
                ray_ptl_trainer,
                param_space={"train_loop_config": train_loop_config},
                run_config=tune_run_config,
                tune_config=ray.tune.TuneConfig(
                    metric="val_acc",
                    mode="max",
                    num_samples=num_trials,
                    scheduler=scheduler,
                    max_concurrent_trials=(2 // lm_params.num_gpus_per_trial),
                ),
            )
            return tuner.fit()
        
        results = tune_asha(num_trials=lm_params.n_trials)
        best_result = results.get_best_result()
        with best_result.checkpoint.as_directory() as checkpoint_dir:
            return Teacher.load_from_checkpoint(f"{checkpoint_dir}/checkpoint.ckpt")
    

    def save_pretrained(self, path: str) -> None:
        """Save the model and tokenizer"""
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

        torch.save(self.model.state_dict(), f"{path}/{self.model_file}")

        with open(f"{path}/config.json", "w") as f:
            cfg = self.train_config.model_dump()
            json.dump({"train_config": cfg}, f, indent=4)

        print(f"Teacher for task {self.train_config.task.task_name} saved to {path}")
        
    @classmethod
    def load_pretrained(cls, path: str) -> 'Teacher':
        with open(f"{path}/config.json", "w") as f:
            hparams = json.load(f)
        
        teacher = cls(**hparams)
        teacher.model = AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype=teacher.train_config.torch_dtype)
        teacher.model_name = path
        teacher.tokenizer = AutoTokenizer.from_pretrained(path)

        print(f"Teacher for task {hparams["task"].name} loaded from {path}")
        return teacher
