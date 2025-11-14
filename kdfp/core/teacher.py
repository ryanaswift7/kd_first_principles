import os
from pydantic import field_validator
import torch
from torch.utils.data import DataLoader
import lightning.pytorch as pl
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

from kdfp.schemas.configBase import ConfigBase
from kdfp.schemas.subConfigs import ExpDirConfig, HyperparamsConfig, LMParamsConfig, TrainingParamsConfig
from kdfp.schemas.tasks import TaskSpec
from kdfp.schemas.mixins import DtypeSerializerMixin, MCEvalMixin
from kdfp.utils.data import load_task_data


def gen_teacher_weights_filename(task_name: str) -> str:
    return f"teacher_{task_name}.bin"

class TeacherModuleConfig(ConfigBase, DtypeSerializerMixin):
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
    num_data_workers: Optional[int] = None
    n_samples: Optional[int] = None

class TeacherModule(pl.LightningModule, MCEvalMixin):
    def __init__(self, train_config: TeacherModuleConfig):
        super().__init__()

        # If train_config is a dict (from loading a checkpoint),
        # parse it back into a TeacherTrainConfig object.
        if isinstance(train_config, dict):
            self.train_config = TeacherModuleConfig(**train_config)
        else:
            # Otherwise (on initial creation), use the object directly.
            self.train_config = train_config
        # self.train_config = train_config

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
        return gen_teacher_weights_filename(self.train_config.task.name)
        
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
        self.log('train_loss', loss, sync_dist=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        preds, true_labels, correct_choice_loss = self._shared_eval_step(
            model=self.model,
            batch=batch,
            batch_idx=batch_idx)
        self.val_metrics.update(preds, true_labels)
        self.log('val_loss', correct_choice_loss, sync_dist=True)
        return {"val_loss": correct_choice_loss}
    
    def on_validation_epoch_end(self):
        self.log_dict(self.val_metrics.compute(), sync_dist=True)
        self.val_metrics.reset()

    def test_step(self, batch, batch_idx):
        preds, true_labels, correct_choice_loss = self._shared_eval_step(
            model=self.model,
            batch=batch,
            batch_idx=batch_idx)
        self.test_metrics.update(preds, true_labels)
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
        
        dataset = dataset.shuffle(seed=self.train_config.seed, buffer_size=1000)
        
        return DataLoader(
            dataset,
            batch_size=self.train_config.batch_size,
            shuffle=False,
            num_workers=self.train_config.num_data_workers,
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
            num_workers=self.train_config.num_data_workers,
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
            num_workers=self.train_config.num_data_workers,
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
        dirs: ExpDirConfig,
    ):
        """Create a teacher for a given task using hyperparameter tuning with Ray Tune"""

        # os.environ["RAY_TRAIN_V2_ENABLED"] = '1'
        # torch.set_float32_matmul_precision('medium')
        pl.seed_everything(training_params.seed, workers=True)
        run_name = f'teacher_{task.name}'
        
        def train_teacher_per_worker(config):
            if training_params.reproducible_backend:
                ray.train.torch.enable_reproducibility()
            train_config = TeacherModuleConfig(
                model_name=lm_name,
                tokenizer_name=lm_name,
                task=task,
                seq_len=training_params.seq_len,
                batch_size=config["batch_size"],
                learning_rate=config["learning_rate"],
                weight_decay=config["weight_decay"],
                max_grad_norm=config["max_grad_norm"],
                n_samples=lm_params.num_samples,
                num_data_workers=training_params.resources.teacher.num_data_workers,
                seed=training_params.seed,
                torch_dtype=training_params.torch_dtype,
            )

            module = TeacherModule(train_config=train_config)
            
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
            module.train()
            trainer.fit(module)
            
        def tune_asha(num_trials: int):
            
            train_loop_config = {
                "learning_rate": ray.tune.qloguniform(hyperparams.lr.mini, hyperparams.lr.maxi, 1e-6),
                "weight_decay": ray.tune.qloguniform(hyperparams.weight_decay.mini, hyperparams.weight_decay.maxi, 1e-4),
                "batch_size": ray.tune.choice(hyperparams.batch_sizes),
                "max_grad_norm": ray.tune.quniform(hyperparams.max_grad_norm.mini, hyperparams.max_grad_norm.maxi, 1e-1)
            }
            scaling_config = ray.train.ScalingConfig(
                num_workers=training_params.resources.teacher.num_ray_workers,
                use_gpu=True,
                resources_per_worker={
                    "CPU": training_params.resources.teacher.num_cpus_per_trial,
                    "GPU": training_params.resources.teacher.num_gpus_per_trial,},
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
                failure_config=ray.tune.FailureConfig(
                    max_failures=training_params.resources.max_failures),
                storage_path=dirs.ray,
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
                    max_concurrent_trials=(training_params.resources.num_available_gpus // training_params.resources.teacher.num_gpus_per_trial),
                ),
            )
            return tuner.fit()
        
        results = tune_asha(num_trials=lm_params.n_trials)
        best_result = results.get_best_result()
        with best_result.checkpoint.as_directory() as checkpoint_dir:
            return TeacherModule.load_from_checkpoint(f"{checkpoint_dir}/checkpoint.ckpt")
    

    def save_pretrained(self, path: str) -> None:
        """Save the model and tokenizer"""
        os.makedirs(path, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        save_path = f"{path}/{self.model_file}"
        torch.save(self.model.state_dict(), save_path)

        with open(f"{path}/pl_config.json", "w") as f:
            # cfg = self.train_config.model_dump()
            # json.dump({"train_config": cfg}, f, indent=4)
            json.dump(self.train_config.model_dump(), f, indent=4)

        print(f"Teacher for task {self.train_config.task.name} saved to {path}")
        
    @classmethod
    def load_pretrained(cls, path: str) -> 'TeacherModule':
        with open(f"{path}/pl_config.json", "r") as f:
            hparams = json.load(f)
        cfg = TeacherModuleConfig(**hparams)
        teacher_module = cls(train_config=cfg)
        teacher_module.model = AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype=teacher_module.train_config.torch_dtype)
        teacher_module.model_name = path
        teacher_module.tokenizer = AutoTokenizer.from_pretrained(path)

        print(f"Teacher for task {teacher_module.train_config.task.name.upper()} loaded from {path}")
        return teacher_module
