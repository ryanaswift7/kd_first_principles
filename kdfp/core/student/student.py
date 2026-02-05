import json
import math
import os
import uuid
from pathlib import Path
from typing import Optional

import lightning.pytorch as pl
import ray
import ray.train
import ray.train.torch
import ray.tune
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from datasets import load_dataset_builder
from lightning.pytorch.loggers import WandbLogger
from ray.air.integrations.wandb import WandbLoggerCallback
from ray.train.lightning import (
    RayDDPStrategy,
    RayLightningEnvironment,
    RayTrainReportCallback,
    prepare_trainer,
)
from ray.tune.schedulers import ASHAScheduler
from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.optuna import OptunaSearch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from transformer_lens import HookedTransformer
from transformers import (
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
)

from kdfp.core.projector.offline.offlineProjector import OfflineProjectorModule
from kdfp.core.projector.online.onlineProjector import OnlineProjector
from kdfp.core.projector.randomized.RandomizedProjector import RandomizedProjector
from kdfp.core.student.training_utils import jensen_shannon_divergence
from kdfp.core.teacher import gen_teacher_weights_filename
from kdfp.schemas.configBase import ConfigBase
from kdfp.schemas.dataset import DatasetConfig
from kdfp.schemas.directories import ExpInstDirs
from kdfp.schemas.experiments.experimentConfig import ExperimentConfig
from kdfp.schemas.hparams import (
    FixedHparamSet,
    HparamsConfig,
    TuneParams,
    get_fixed_student_params,
)
from kdfp.schemas.mixins import DtypeSerializerMixin, MCEvalMixin
from kdfp.schemas.models import DistillationPair
from kdfp.schemas.objective_weights import ObjectiveWeights
from kdfp.schemas.projections import ProjectionInstance
from kdfp.schemas.trainingParams import TrainingParams
from kdfp.utils.data import load_data
from kdfp.utils.enums import ProjectionDirection, ProjectionTraining, TrainingType
from kdfp.utils.shallow_model_loading import load_shallow
from kdfp.utils.weight_conversion import (
    convert_hooked_to_neox_weights,
    convert_neox_to_hooked_weights,
)


def gen_student_weights_filename() -> str:
    return "student.bin"


class StudentModuleConfig(ConfigBase, DtypeSerializerMixin):
    exp_name: str
    distill_pair: DistillationPair
    dirs: ExpInstDirs
    projection: Optional[ProjectionInstance] = None
    dataset: DatasetConfig
    seq_len: int
    batch_size: int
    temperature: float
    learning_rate: float
    weight_decay: float
    max_grad_norm: float
    seed: int
    torch_dtype: torch.dtype
    num_data_workers: int = 0
    total_tokens: Optional[int]
    num_samples: Optional[int]
    objective_weights: Optional[ObjectiveWeights]
    jsd_beta: Optional[float]
    total_steps: int
    finetune_teacher: bool
    training_type: TrainingType
    streaming: bool


class StudentModule(pl.LightningModule, MCEvalMixin):
    def __init__(self, train_config):
        super().__init__()

        # If train_config is a dict (from loading a checkpoint),
        # parse it back into a TeacherTrainConfig object.
        if isinstance(train_config, dict):
            self.train_config = StudentModuleConfig(**train_config)
        else:
            # Otherwise (on initial creation), use the object directly.
            self.train_config = train_config
        # self.train_config = train_config

        # Now self.train_config is guaranteed to be a TeacherTrainConfig object,
        # and you can safely call .model_dump()
        self.save_hyperparameters({"train_config": self.train_config.model_dump()})

        # self.tokenizer = AutoTokenizer.from_pretrained(self.train_config.tokenizer_name)
        self.teacher = self.load_teacher()
        self.student = self.load_student()
        self.tokenizer = self.student.tokenizer
        self.projector = self.load_projector()

        # Add padding token if it doesn't exist
        assert self.tokenizer is not None, "Tokenizer could not be loaded."
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Data collator for language modeling
        self.data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # Causal LM, not masked LM
            return_tensors="pt",
        )
        self.mse = nn.MSELoss()

        # self.val_metrics = torchmetrics.MetricCollection(
        #     {"ppl": torchmetrics.Perplexity(ignore_index=self.tokenizer.eos_token_id)},
        #     prefix="val_",
        # )
        # self.test_metrics = self.val_metrics.clone(prefix="test_")

    def load_teacher(self) -> HookedTransformer:
        if self.train_config.finetune_teacher:
            teacher_hf = AutoModelForCausalLM.from_pretrained(
                f"{self.train_config.dirs.teacher_model}/{gen_teacher_weights_filename()}",
            )
            # load base teacher
            teacher_tl = HookedTransformer.from_pretrained_no_processing(
                self.train_config.distill_pair.teacher.name,
                dtype=self.train_config.torch_dtype,
            )

            print("Loading finetuned weights for teacher model...")
            # load finetuned weights, convert to transformer lens format, load into teacher
            tl_teacher_weights = convert_neox_to_hooked_weights(
                neox_model=teacher_hf, cfg=teacher_tl.cfg
            )
            teacher_tl.load_state_dict(tl_teacher_weights, strict=False)
        else:
            teacher_tl = HookedTransformer.from_pretrained_no_processing(
                self.train_config.distill_pair.teacher.name,
                dtype=self.train_config.torch_dtype,
            )
        return teacher_tl.eval()

    def load_student(self) -> HookedTransformer:
        if self.train_config.distill_pair.student.layer_reduction_factor:
            student, tokenizer, keep_indices = load_shallow(
                model_name=self.train_config.distill_pair.student.name,
                mode="tl",
                layer_reduction_factor=self.train_config.distill_pair.student.layer_reduction_factor,
                selection_method="linspace",
                torch_dtype=self.train_config.torch_dtype,
            )
        else:
            student = HookedTransformer.from_pretrained_no_processing(
                self.train_config.distill_pair.student.name,
                dtype=self.train_config.torch_dtype,
            )
        return student

    def load_projector(self) -> torch.nn.Module:
        if self.train_config.projection:
            if (
                self.train_config.projection.training_method
                == ProjectionTraining.OFFLINE
            ):
                # weights_file = Path(self.train_config.dirs.off_proj_models)
                p = OfflineProjectorModule.load_pretrained(
                    f"{self.train_config.dirs.off_proj_models}"
                )
                # + f"/{gen_off_proj_weights_filename(self.train_config.projection.offline_projector_alignment_beta)}")
                return p.extract_single_direction(
                    self.train_config.projection.direction
                ).eval()
            elif (
                self.train_config.projection.training_method
                == ProjectionTraining.ONLINE
            ):
                return OnlineProjector(
                    teacher_dim=self.train_config.distill_pair.teacher.d_model,
                    student_dim=self.train_config.distill_pair.student.d_model,
                    direction=self.train_config.projection.direction,
                    torch_dtype=self.train_config.torch_dtype,
                ).train()
            elif (
                self.train_config.projection.training_method
                == ProjectionTraining.RANDOMIZED
            ):
                return RandomizedProjector(
                    teacher_dim=self.train_config.distill_pair.teacher.d_model,
                    student_dim=self.train_config.distill_pair.student.d_model,
                    direction=self.train_config.projection.direction,
                    torch_dtype=self.train_config.torch_dtype,
                ).eval()
            else:
                raise ValueError(
                    f"Unrecognized projection training method: {self.train_config.projection.training_method.value.upper()}"
                )
        else:
            return nn.Identity()

    def forward(self, input_ids, attention_mask, labels=None):
        return self.student(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]

        student_hooks = self.train_config.distill_pair.student.hook_list
        teacher_hooks = self.train_config.distill_pair.teacher.hook_list
        obj_weights = self.train_config.objective_weights
        assert obj_weights is not None, (
            "Objective weights must be provided for training."
        )

        total_loss = 0

        if teacher_hooks:
            n = len(teacher_hooks)
            assert n == len(student_hooks)

            student_output, student_cache = self.student.run_with_cache(
                input_ids,
                return_type="both",
                names_filter=lambda name: name in student_hooks,
            )

            with torch.no_grad():
                teacher_output, teacher_cache = self.teacher.run_with_cache(
                    input_ids,
                    return_type="both",
                    names_filter=lambda name: name in teacher_hooks,
                )
        else:
            student_output = self.student(input_ids, return_type="both")

            with torch.no_grad():
                teacher_output = self.teacher(input_ids, return_type="both")

        total_loss += obj_weights.true_labels * student_output.loss

        # finetuning => no distillation
        if self.train_config.training_type == TrainingType.FT:
            self.log("train_loss", total_loss, sync_dist=True)
            return total_loss

        # distill outputs
        assert self.train_config.jsd_beta is not None, (
            "JSD beta must be set for KD training."
        )
        total_loss += obj_weights.teacher_outputs * jensen_shannon_divergence(
            teacher_logits=teacher_output.logits,
            student_logits=student_output.logits,
            temperature=self.train_config.temperature,
            beta=self.train_config.jsd_beta,
        )

        # distill internal acts
        act_substrings_no_attn_scores = ["embed", "attn_out", "mlp", "resid"]
        for i in range(n):
            # handle attn scores as a distribution
            if "attn_scores" in student_hooks[i]:
                assert "attn_scores" in teacher_hooks[i], (
                    f"Expected teacher to be an 'attn_scores' activation, but found {teacher_hooks[i]}"
                )
                weight = getattr(obj_weights, "attn_scores")
                total_loss += weight * jensen_shannon_divergence(
                    teacher_logits=teacher_cache[teacher_hooks[i]],
                    student_logits=student_cache[student_hooks[i]],
                    temperature=self.train_config.temperature,
                    beta=self.train_config.jsd_beta,
                )
            # handle the rest directly as regular activations
            for act in act_substrings_no_attn_scores:
                if act in student_hooks[i]:
                    assert act in teacher_hooks[i], (
                        f"Expected to find teacher's '{act}' activation, but found {teacher_hooks[i]}"
                    )
                    weight = getattr(obj_weights, act)
                    # print(f"Grabbed activation: {act}")
                    if self.train_config.projection:
                        if (
                            self.train_config.projection.direction
                            == ProjectionDirection.UP
                        ):
                            raw_loss = self.mse(
                                teacher_cache[teacher_hooks[i]],
                                self.projector(student_cache[student_hooks[i]]),
                            )
                        elif (
                            self.train_config.projection.direction
                            == ProjectionDirection.DOWN
                        ):
                            raw_loss = self.mse(
                                self.projector(teacher_cache[teacher_hooks[i]]),
                                student_cache[student_hooks[i]],
                            )
                        else:
                            raise ValueError(
                                f"Unrecognized projection direction: {self.train_config.projection.direction.value.upper()}"
                            )
                    else:
                        raw_loss = self.mse(
                            teacher_cache[teacher_hooks[i]],
                            student_cache[student_hooks[i]],
                        )
                    total_loss += weight * raw_loss
                    break  # go back to outer loop and process next hook

        self.log("train_loss", total_loss, sync_dist=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        target_device = next(self.student.parameters()).device
        batch = batch.to(target_device)
        logits = self.student(batch["input_ids"], return_type="logits")

        # 2. Align Logits and Labels
        # TransformerLens logits are [batch, seq_len, vocab_size]
        # We need to predict the NEXT token, so we shift:

        # Shift logits: Remove the last token's prediction (it has no target)
        shift_logits = logits[..., :-1, :].contiguous()

        # Shift labels: Remove the first token (it has no history to predict it from)
        shift_labels = batch["labels"][..., 1:].contiguous()

        # 3. Calculate Cross Entropy manually
        # 'ignore_index=-100' is the magic key that prevents NaNs from padding
        val_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )

        perplexity = torch.exp(val_loss)

        self.log("val_loss", val_loss, prog_bar=True)
        self.log("val_ppl", perplexity, prog_bar=True)

        return val_loss

    # def on_validation_epoch_end(self):
    #     self.log_dict(self.val_metrics.compute(), sync_dist=True)
    #     self.val_metrics.reset()

    def test_step(self, batch, batch_idx):
        target_device = next(self.student.parameters()).device
        batch = batch.to(target_device)

        logits = self.student(batch["input_ids"], return_type="logits")

        # 2. Align Logits and Labels
        # TransformerLens logits are [batch, seq_len, vocab_size]
        # We need to predict the NEXT token, so we shift:

        # Shift logits: Remove the last token's prediction (it has no target)
        shift_logits = logits[..., :-1, :].contiguous()

        # Shift labels: Remove the first token (it has no history to predict it from)
        shift_labels = batch["labels"][..., 1:].contiguous()

        # 3. Calculate Cross Entropy manually
        # 'ignore_index=-100' is the magic key that prevents NaNs from padding
        test_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )

        perplexity = torch.exp(test_loss)

        self.log("test_loss", test_loss, prog_bar=True)
        self.log("test_ppl", perplexity, prog_bar=True)

        return test_loss

    # def on_test_epoch_end(self):
    #     self.log_dict(self.test_metrics.compute(), sync_dist=True)
    #     self.test_metrics.reset()

    def configure_optimizers(self):
        if isinstance(self.projector, OnlineProjector):
            optimizer = torch.optim.AdamW(
                [
                    {"params": self.student.parameters()},
                    {"params": self.projector.parameters()},
                ],
                lr=self.train_config.learning_rate,
                weight_decay=self.train_config.weight_decay,
            )
        else:
            optimizer = torch.optim.AdamW(
                self.student.parameters(),
                lr=self.train_config.learning_rate,
                weight_decay=self.train_config.weight_decay,
            )
            if self.projector:
                self.projector.eval()

        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.train_config.total_steps,
            eta_min=1e-9,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def train_dataloader(self):
        dataset = load_data(
            dataset_name=self.train_config.dataset.name,
            subset=self.train_config.dataset.subset,
            tokenizer=self.tokenizer,
            seq_len=self.train_config.seq_len,
            split="train",
            n_samples=self.train_config.num_samples,
            streaming=self.train_config.streaming,
        )

        dataset = dataset.shuffle(seed=self.train_config.seed, buffer_size=1000)

        return DataLoader(
            dataset,
            batch_size=self.train_config.batch_size,
            shuffle=False,
            num_workers=self.train_config.num_data_workers,
            collate_fn=self.data_collator,
            pin_memory=True,
        )

    # def val_dataloader(self):
    #     dataset = load_data(
    #         dataset_name=self.train_config.dataset.name,
    #         subset=self.train_config.dataset.subset,
    #         tokenizer=self.tokenizer,
    #         seq_len=self.train_config.seq_len,
    #         split="validation",
    #         n_samples=self.train_config.n_samples,
    #         streaming=self.train_config.streaming,
    #     )

    #     return DataLoader(
    #         dataset,
    #         batch_size=self.train_config.batch_size,
    #         shuffle=False,
    #         num_workers=self.train_config.num_data_workers,
    #         collate_fn=self.data_collator,
    #         pin_memory=True,
    #     )

    # def test_dataloader(self):
    #     dataset = load_data(
    #         dataset_name=self.train_config.dataset.name,
    #         subset=self.train_config.dataset.subset,
    #         tokenizer=self.tokenizer,
    #         seq_len=self.train_config.seq_len,
    #         split="test",
    #         n_samples=self.train_config.n_samples,
    #         streaming=self.train_config.streaming,
    #     )

    #     return DataLoader(
    #         dataset,
    #         batch_size=self.train_config.batch_size,
    #         shuffle=False,
    #         num_workers=self.train_config.num_data_workers,
    #         collate_fn=self.data_collator,
    #         pin_memory=True,
    #     )

    @classmethod
    def create(
        cls,
        cfg: ExperimentConfig,
        dirs: ExpInstDirs,
        hyperparams: HparamsConfig,
        training_params: TrainingParams,
    ):
        pl.seed_everything(cfg.seed, workers=True)

        if cfg.training_type == TrainingType.NO_TRAIN:
            no_train_param = -1
            no_train_config = StudentModuleConfig(
                exp_name=cfg.name,
                distill_pair=cfg.distill_pair,
                dirs=dirs,
                projection=cfg.projection,
                dataset=training_params.dataset,
                seq_len=training_params.seq_len,
                batch_size=training_params.train_batch_size,
                learning_rate=no_train_param,
                weight_decay=no_train_param,
                max_grad_norm=no_train_param,
                temperature=no_train_param,
                seed=cfg.seed,
                torch_dtype=training_params.torch_dtype,
                num_data_workers=training_params.resources.student.num_data_workers,
                total_tokens=training_params.total_tokens,
                jsd_beta=None,
                objective_weights=None,
                total_steps=no_train_param,
                num_samples=no_train_param,
                finetune_teacher=training_params.controls.finetune_teacher,
                training_type=cfg.training_type,
                streaming=training_params.resources.stream_task_data,
            )
            return StudentModule(train_config=no_train_config)

        run_name = f"student_{cfg.name}"
        builder = load_dataset_builder(
            training_params.dataset.name, training_params.dataset.subset
        )
        n_total_train_samples = builder.info.splits["train"].num_examples  # type: ignore
        n_train_samples = (
            min(n_total_train_samples, training_params.num_samples)
            if training_params.num_samples
            else n_total_train_samples
        )
        total_steps = math.ceil(
            (n_train_samples * training_params.num_epochs)
            / training_params.train_batch_size
        )

        def train_student_per_worker(config):
            if training_params.reproducible_backend:
                ray.train.torch.enable_reproducibility()

            assert cfg.objective is not None, (
                "Objective config must be provided if training."
            )
            if cfg.training_type == TrainingType.KD:
                assert cfg.objective.jsd_beta is not None, (
                    "JSD beta must be set for KD training."
                )
            train_config = StudentModuleConfig(
                exp_name=cfg.name,
                distill_pair=cfg.distill_pair,
                dirs=dirs,
                projection=cfg.projection,
                dataset=training_params.dataset,
                seq_len=training_params.seq_len,
                batch_size=training_params.train_batch_size,
                learning_rate=config["learning_rate"],
                weight_decay=config["weight_decay"],
                max_grad_norm=config["max_grad_norm"],
                temperature=config["temperature"],
                seed=cfg.seed,
                torch_dtype=training_params.torch_dtype,
                num_data_workers=training_params.resources.student.num_data_workers,
                total_tokens=training_params.total_tokens,
                jsd_beta=cfg.objective.jsd_beta,
                objective_weights=cfg.objective.weights,
                total_steps=total_steps,
                num_samples=n_train_samples,
                finetune_teacher=training_params.controls.finetune_teacher,
                training_type=cfg.training_type,
                streaming=training_params.resources.stream_task_data,
            )

            module = StudentModule(train_config=train_config)

            trainer = pl.Trainer(
                devices="auto",
                accelerator="auto",
                strategy=RayDDPStrategy(find_unused_parameters=True),
                plugins=[RayLightningEnvironment()],
                callbacks=[RayTrainReportCallback()],
                max_epochs=training_params.num_epochs,
                gradient_clip_algorithm="norm",
                gradient_clip_val=config["max_grad_norm"],
                enable_progress_bar=False,
                enable_checkpointing=False,
                deterministic=training_params.reproducible_backend,
                check_val_every_n_epoch=1,
                logger=WandbLogger(
                    name=config["wandb_run_name"],
                    project=training_params.wandb.project,
                    group=training_params.wandb.group,
                ),
            )
            try:
                wandb.init(
                    project=training_params.wandb.project,
                    group=training_params.wandb.group,
                    name=config["wandb_run_name"],
                    reinit=True,
                )
                trainer = prepare_trainer(trainer)
                module.train()
                trainer.fit(module)
            finally:
                wandb.finish()

        def train_fixed_params(fixed_params: FixedHparamSet):
            fixed_config = {
                "learning_rate": fixed_params.lr,
                "weight_decay": fixed_params.weight_decay,
                "max_grad_norm": fixed_params.max_grad_norm,
                "temperature": fixed_params.temperature,
                "wandb_run_name": f"{run_name}_fixed_{uuid.uuid4().hex[:6]}",
            }
            scaling_config = ray.train.ScalingConfig(
                num_workers=training_params.resources.student.num_ray_workers,
                use_gpu=True,
                resources_per_worker={
                    "CPU": training_params.resources.student.num_cpus_per_trial,
                    "GPU": training_params.resources.student.num_gpus_per_trial,
                },
                placement_strategy="PACK",
            )
            train_run_config = ray.train.RunConfig(
                name=f"{run_name}_fixed_{uuid.uuid4().hex[:6]}",
                checkpoint_config=ray.tune.CheckpointConfig(
                    num_to_keep=1,
                    checkpoint_frequency=0,
                    checkpoint_score_attribute="val_ppl",
                    checkpoint_score_order="min",
                ),
                failure_config=ray.train.FailureConfig(
                    max_failures=training_params.resources.max_failures
                ),
                storage_path=str(dirs.ray),
            )
            trainer = ray.train.torch.TorchTrainer(
                train_student_per_worker,
                train_loop_config=fixed_config,
                run_config=train_run_config,
                scaling_config=scaling_config,
            )

            return trainer.fit()

        def tune_asha(num_trials: int, tune_params: TuneParams):
            tune_config = {
                "learning_rate": ray.tune.qloguniform(
                    tune_params.lr.mini, tune_params.lr.maxi, 1e-6
                ),
                "weight_decay": ray.tune.qloguniform(
                    tune_params.weight_decay.mini, tune_params.weight_decay.maxi, 1e-4
                ),
                "max_grad_norm": ray.tune.quniform(
                    tune_params.max_grad_norm.mini, tune_params.max_grad_norm.maxi, 1e-1
                ),
                "temperature": ray.tune.quniform(
                    tune_params.temperature.mini, tune_params.temperature.maxi, 1e-1
                ),
                "wandb_run_name": f"{run_name}_tune_{uuid.uuid4().hex[:6]}",
            }
            scaling_config = ray.train.ScalingConfig(
                num_workers=training_params.resources.student.num_ray_workers,
                use_gpu=True,
                resources_per_worker={
                    "CPU": training_params.resources.student.num_cpus_per_trial,
                    "GPU": training_params.resources.student.num_gpus_per_trial,
                },
                placement_strategy="PACK",
            )
            tune_run_config = ray.tune.RunConfig(
                name=tune_config["wandb_run_name"],
                checkpoint_config=ray.tune.CheckpointConfig(
                    num_to_keep=1,
                    checkpoint_frequency=0,
                    checkpoint_score_attribute="val_ppl",
                    checkpoint_score_order="min",
                ),
                failure_config=ray.tune.FailureConfig(
                    max_failures=training_params.resources.max_failures
                ),
                storage_path=str(dirs.ray),
                callbacks=[
                    WandbLoggerCallback(
                        project=training_params.wandb.project,
                        group=training_params.wandb.group,
                        run_name=tune_config["wandb_run_name"],
                    )
                ],
            )
            ray_ptl_trainer = ray.train.torch.TorchTrainer(
                train_student_per_worker,
                scaling_config=scaling_config,
            )

            optuna_search = OptunaSearch(
                metric="val_ppl",  # Preserving your optimization goal
                mode="min",
                seed=cfg.seed,
            )

            search_alg = ConcurrencyLimiter(
                optuna_search,
                max_concurrent=math.floor(
                    training_params.resources.num_available_gpus
                    / training_params.resources.student.num_gpus_per_trial
                ),
            )

            asha_scheduler = ASHAScheduler(
                time_attr="training_iteration",
                max_t=total_steps,
                grace_period=1,  # Minimum number of iterations before pruning can occur
                reduction_factor=3,
            )
            tuner = ray.tune.Tuner(
                trainable=ray_ptl_trainer,
                param_space={"train_loop_config": tune_config},
                run_config=tune_run_config,
                tune_config=ray.tune.TuneConfig(
                    metric="val_ppl",
                    mode="min",
                    num_samples=num_trials,
                    scheduler=asha_scheduler,
                    search_alg=search_alg,
                ),
            )
            return tuner.fit()

        fixed_params = get_fixed_student_params(
            hyperparams, cfg.student_type, cfg.training_type
        )

        if fixed_params:
            best_result = train_fixed_params(fixed_params=fixed_params)
        elif hyperparams.tune:
            best_result = tune_asha(
                num_trials=training_params.n_trials, tune_params=hyperparams.tune
            ).get_best_result()
        else:
            raise ValueError("Neither valid fixed params nor tuning params were found.")

        assert best_result.checkpoint is not None, (
            "No checkpoint found in best Student result."
        )
        with best_result.checkpoint.as_directory() as checkpoint_dir:
            return StudentModule.load_from_checkpoint(
                f"{checkpoint_dir}/checkpoint.ckpt"
            )

    @property
    def model_file(self) -> str:
        return gen_student_weights_filename()

    def save_pretrained(self, path: str):
        os.makedirs(path, exist_ok=True)
        student_tl_state_dict = self.student.state_dict()
        save_path = f"{path}/{self.model_file}"
        torch.save(student_tl_state_dict, save_path)

        student_hf_state_dict, _ = convert_hooked_to_neox_weights(
            self.student, cfg=self.student.cfg
        )
        if self.train_config.distill_pair.student.layer_reduction_factor:
            hf, _, _ = load_shallow(
                model_name=self.train_config.distill_pair.student.name,
                mode="hf",
                layer_reduction_factor=self.train_config.distill_pair.student.layer_reduction_factor,
                selection_method="linspace",
                torch_dtype=self.train_config.torch_dtype,
            )
        else:
            hf = AutoModelForCausalLM.from_pretrained(
                self.train_config.distill_pair.student.name,
                dtype=self.train_config.torch_dtype,
            )
        hf.load_state_dict(student_hf_state_dict)
        hf.save_pretrained(path)  # type: ignore
        self.tokenizer.save_pretrained(path)  # type: ignore

        with open(f"{path}/pl_config.json", "w") as f:
            json.dump(self.train_config.model_dump(), f, indent=4)

        print(f"Student saved to {path}")

    @classmethod
    def load_pretrained(cls, path: str) -> "StudentModule":
        with open(f"{path}/pl_config.json", "r") as f:
            hparams = json.load(f)
        cfg = StudentModuleConfig(**hparams)
        student_module = cls(train_config=cfg)
        student_tl_state_dict = torch.load(
            Path(path) / student_module.model_file, weights_only=True
        )
        student_module.student.load_state_dict(student_tl_state_dict)
        print(f"Student loaded from {path}")
        return student_module
