import json
import os
import uuid
from pathlib import Path
from typing import Optional, Union

import lightning.pytorch as pl
import ray
import torch
import torchmetrics
from lightning.pytorch.loggers import WandbLogger
from ray.train.lightning import (
    RayDDPStrategy,
    RayLightningEnvironment,
    RayTrainReportCallback,
    prepare_trainer,
)
from ray.tune.schedulers import ASHAScheduler
from torch.utils.data import DataLoader
from torchmetrics import Accuracy, F1Score
from transformer_lens import HookedTransformer
from transformers import (
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
)

from kdfp.core.projector.offline.offlineProjector import OfflineProjectorModule
from kdfp.core.projector.online.onlineProjector import OnlineProjector
from kdfp.core.projector.randomized.RandomizedProjector import RandomizedProjector
from kdfp.core.student.training_utils import jensen_shannon_divergence
from kdfp.schemas.configBase import ConfigBase
from kdfp.schemas.lmConfig import LMConfig
from kdfp.schemas.mixins import DtypeSerializerMixin, MCEvalMixin
from kdfp.schemas.objective_weights import ObjectiveWeights
from kdfp.schemas.tasks import TaskInstance
from kdfp.utils.data import load_task_data
from kdfp.utils.enums import ProjectionDirection, ProjectionTraining
from kdfp.utils.shallow_model_loading import load_shallow
from kdfp.utils.weight_conversion import (
    convert_hooked_to_neox_weights,
    convert_neox_to_hooked_weights,
)


def gen_student_weights_filename(task_name: str) -> str:
    return f"student_{task_name}.bin"


class StudentModuleConfig(ConfigBase, DtypeSerializerMixin):
    student_cfg: LMConfig
    teacher_cfg: LMConfig
    dirs: ExpDirConfig
    projection: Optional[ProjectionConfig] = None
    task: TaskInstance
    seq_len: int
    batch_size: int
    temperature: float
    learning_rate: float
    weight_decay: float
    max_grad_norm: float
    seed: int
    torch_dtype: Union[str, torch.dtype]
    num_data_workers: Optional[int] = None
    n_samples: Optional[int] = None
    objective_weights: ObjectiveWeights
    jsd_beta: Optional[float]


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
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Data collator for language modeling
        self.data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # Causal LM, not masked LM
            return_tensors="pt",
        )

        if self.train_config.task.task_type == "multiclass":
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
            prefix="val_",
        )
        self.test_metrics = self.val_metrics.clone(prefix="test_")

        # self.train_losses = []

    def load_teacher(self) -> HookedTransformer:
        teacher_hf = AutoModelForCausalLM.from_pretrained(
            f"{self.train_config.dirs.teacher_model}/{self.train_config.task.name}"
        )
        # load base teacher
        teacher_tl = HookedTransformer.from_pretrained_no_processing(
            self.train_config.teacher_cfg.model_name,
            dtype=self.train_config.torch_dtype,
        )

        # load finetuned weights, convert to transformer lens format, load into teacher
        # weights_file = Path(self.train_config.dirs.teacher_models) / gen_teacher_weights_filename(task_name=self.train_config.task.name)
        # teacher_weights = torch.load(weights_file, weights_only=True)
        # print("\n" + "="*60)
        # print(teacher_weights)
        # print("="*60)
        tl_teacher_weights = convert_neox_to_hooked_weights(
            neox_model=teacher_hf, cfg=teacher_tl.cfg
        )
        teacher_tl.load_state_dict(tl_teacher_weights, strict=False)
        # teacher_tl.eval()
        return teacher_tl.eval()

    def load_student(self) -> HookedTransformer:
        if self.train_config.student_cfg.layer_reduction_factor:
            student, tokenizer, keep_indices = load_shallow(
                model_name=self.train_config.student_cfg.model_name,
                mode="tl",
                layer_reduction_factor=self.train_config.student_cfg.layer_reduction_factor,
                selection_method="linspace",
                torch_dtype=self.train_config.torch_dtype,
            )
        else:
            student = HookedTransformer.from_pretrained_no_processing(
                self.train_config.student_cfg.model_name,
                dtype=self.train_config.torch_dtype,
            )
        return student

    def load_projector(self):
        if self.train_config.projection:
            if self.train_config.projection.training == ProjectionTraining.OFFLINE:
                # weights_file = Path(self.train_config.dirs.off_proj_models)
                p = OfflineProjectorModule.load_pretrained(
                    f"{self.train_config.dirs.off_proj_models}"
                )
                # + f"/{gen_off_proj_weights_filename(self.train_config.projection.offline_projector_alignment_beta)}")
                return p.extract_single_direction(
                    self.train_config.projection.direction
                ).eval()
            elif self.train_config.projection.training == ProjectionTraining.ONLINE:
                return OnlineProjector(
                    teacher_dim=self.train_config.teacher_cfg.d_model,
                    student_dim=self.train_config.student_cfg.d_model,
                    direction=self.train_config.projection.direction,
                    torch_dtype=self.train_config.torch_dtype,
                ).train()
            elif self.train_config.projection.training == ProjectionTraining.RANDOMIZED:
                return RandomizedProjector(
                    teacher_dim=self.train_config.teacher_cfg.d_model,
                    student_dim=self.train_config.student_cfg.d_model,
                    direction=self.train_config.projection.direction,
                    torch_dtype=self.train_config.torch_dtype,
                ).eval()
        else:
            # self.projector = IdentityProjector()
            return None

    @property
    def model_file(self):
        return gen_student_weights_filename(self.train_config.task.name)

    def forward(self, input_ids, attention_mask, labels=None):
        return self.student(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )

    def training_step(self, batch, batch_idx):
        mse = torch.nn.MSELoss()
        student_hooks = self.train_config.student_cfg.hook_list
        teacher_hooks = self.train_config.teacher_cfg.hook_list
        obj_weights = self.train_config.objective_weights

        input_ids = batch["input_ids"]
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

        # no JSD beta => finetuning => no distillation
        if self.train_config.jsd_beta is None:
            self.log("train_loss", total_loss, sync_dist=True)
            return total_loss

        # distill outputs
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
                            raw_loss = mse(
                                teacher_cache[teacher_hooks[i]],
                                self.projector(student_cache[student_hooks[i]]),
                            )
                        elif (
                            self.train_config.projection.direction
                            == ProjectionDirection.DOWN
                        ):
                            raw_loss = mse(
                                self.projector(teacher_cache[teacher_hooks[i]]),
                                student_cache[student_hooks[i]],
                            )
                        else:
                            raise ValueError(
                                f"Unrecognized projection direction: {self.train_config.projection.direction.value.upper()}"
                            )
                    else:
                        raw_loss = mse(
                            teacher_cache[teacher_hooks[i]],
                            student_cache[student_hooks[i]],
                        )
                    total_loss += weight * raw_loss
                    break  # go back to outer loop and process next hook

        self.log("train_loss", total_loss, sync_dist=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        preds, true_labels, correct_choice_loss = self._shared_eval_step(
            model=self.student, batch=batch, batch_idx=batch_idx
        )
        self.val_metrics.update(preds, true_labels)
        self.log("val_loss", correct_choice_loss, sync_dist=True)
        return {"val_loss": correct_choice_loss}

    def on_validation_epoch_end(self):
        self.log_dict(self.val_metrics.compute(), sync_dist=True)
        self.val_metrics.reset()

    def test_step(self, batch, batch_idx):
        preds, true_labels, correct_choice_loss = self._shared_eval_step(
            model=self.student, batch=batch, batch_idx=batch_idx
        )
        self.test_metrics.update(preds, true_labels)
        self.log("test_loss", correct_choice_loss, sync_dist=True)
        return {"test_loss": correct_choice_loss}

    def on_test_epoch_end(self):
        self.log_dict(self.test_metrics.compute(), sync_dist=True)
        self.test_metrics.reset()

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
        return optimizer

    def train_dataloader(self):
        dataset = load_task_data(
            task=self.train_config.task,
            tokenizer=self.tokenizer,
            seq_len=self.train_config.seq_len,
            split="train",
            n_samples=self.train_config.n_samples,
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

    # NOTE: can't use self.data_collator bc we tokenized in a custom format
    def val_dataloader(self):
        dataset = load_task_data(
            task=self.train_config.task,
            tokenizer=self.tokenizer,
            seq_len=self.train_config.seq_len,
            split="validation",
            n_samples=self.train_config.n_samples,
        )

        return DataLoader(
            dataset,
            batch_size=self.train_config.batch_size,
            shuffle=False,
            num_workers=self.train_config.num_data_workers,
            # collate_fn=self.data_collator,
            pin_memory=True,
        )

    def test_dataloader(self):
        dataset = load_task_data(
            task=self.train_config.task,
            tokenizer=self.tokenizer,
            seq_len=self.train_config.seq_len,
            split="test",
            n_samples=self.train_config.n_samples,
        )

        return DataLoader(
            dataset,
            batch_size=self.train_config.batch_size,
            shuffle=False,
            num_workers=self.train_config.num_data_workers,
            # collate_fn=self.data_collator,
            pin_memory=True,
        )

    @classmethod
    def create(
        cls,
        exp_name: str,
        student_cfg: LMConfig,
        teacher_cfg: LMConfig,
        dirs: ExpDirConfig,
        task: TaskInstance,
        lm_params: LMParamsConfig,
        hyperparams: HyperparamsConfig,
        training_params: TrainingParamsConfig,
        projection: ProjectionConfig,
        jsd_beta: Optional[float],
        objective_weights: ObjectiveWeights,
    ):
        pl.seed_everything(training_params.seed, workers=True)
        run_name = f"student_{exp_name}"

        def train_student_per_worker(config):
            if training_params.reproducible_backend:
                ray.train.torch.enable_reproducibility()
            train_config = StudentModuleConfig(
                student_cfg=student_cfg,
                teacher_cfg=teacher_cfg,
                dirs=dirs,
                projection=projection,
                task=task,
                seq_len=training_params.seq_len,
                batch_size=config["batch_size"],
                learning_rate=config["learning_rate"],
                weight_decay=config["weight_decay"],
                max_grad_norm=config["max_grad_norm"],
                temperature=config["temperature"],
                seed=training_params.seed,
                torch_dtype=training_params.torch_dtype,
                num_data_workers=training_params.resources.student.num_data_workers,
                n_samples=lm_params.num_samples,
                jsd_beta=jsd_beta,
                objective_weights=objective_weights,
            )

            module = StudentModule(train_config=train_config)

            trainer = pl.Trainer(
                devices="auto",
                accelerator="auto",
                strategy=RayDDPStrategy(find_unused_parameters=True),
                plugins=[RayLightningEnvironment()],
                callbacks=[RayTrainReportCallback()],
                max_epochs=lm_params.num_epochs,
                gradient_clip_algorithm="norm",
                gradient_clip_val=config["max_grad_norm"],
                enable_progress_bar=False,
                enable_checkpointing=False,
                deterministic=training_params.reproducible_backend,
                check_val_every_n_epoch=1,
                logger=WandbLogger(
                    name=f"{run_name}_tune_{uuid.uuid4().hex[:6]}",
                    project="kdfp",
                    group="jsd_beta_search",
                ),
            )
            trainer = prepare_trainer(trainer)
            module.train()
            trainer.fit(module)

        def tune_asha(num_trials: int):
            train_loop_config = {
                "learning_rate": ray.tune.qloguniform(
                    hyperparams.lr.mini, hyperparams.lr.maxi, 1e-6
                ),
                "weight_decay": ray.tune.qloguniform(
                    hyperparams.weight_decay.mini, hyperparams.weight_decay.maxi, 1e-4
                ),
                "batch_size": ray.tune.choice(hyperparams.batch_sizes),
                "max_grad_norm": ray.tune.quniform(
                    hyperparams.max_grad_norm.mini, hyperparams.max_grad_norm.maxi, 1e-1
                ),
                "temperature": ray.tune.quniform(
                    hyperparams.temperature.mini, hyperparams.temperature.maxi, 1e-1
                ),
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
                name=f"{run_name}_tune_{uuid.uuid4().hex[:6]}",
                checkpoint_config=ray.tune.CheckpointConfig(
                    num_to_keep=1,
                    checkpoint_frequency=0,
                    checkpoint_score_attribute="val_acc",
                    checkpoint_score_order="max",
                ),
                failure_config=ray.tune.FailureConfig(
                    max_failures=training_params.resources.max_failures
                ),
                storage_path=dirs.ray,
                # callbacks=[WandbLoggerCallback(
                #     project="kdfp",
                #     group="students",
                #     run_name=f"{run_name}_tune_{uuid.uuid4().hex[:6]}",
                # )]
            )
            ray_ptl_trainer = ray.train.torch.TorchTrainer(
                train_student_per_worker,
                scaling_config=scaling_config,
            )

            scheduler = ASHAScheduler(
                max_t=lm_params.num_epochs, grace_period=1, reduction_factor=2
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
                    max_concurrent_trials=(
                        training_params.resources.num_available_gpus
                        // training_params.resources.student.num_gpus_per_trial
                    ),
                ),
            )
            return tuner.fit()

        results = tune_asha(num_trials=lm_params.n_trials)
        best_result = results.get_best_result()
        with best_result.checkpoint.as_directory() as checkpoint_dir:
            return StudentModule.load_from_checkpoint(
                f"{checkpoint_dir}/checkpoint.ckpt"
            )

    def save_pretrained(self, path: str):
        os.makedirs(path, exist_ok=True)
        student_tl_state_dict = self.student.state_dict()
        save_path = f"{path}/{self.model_file}"
        torch.save(student_tl_state_dict, save_path)

        student_hf_state_dict, _ = convert_hooked_to_neox_weights(
            self.student, cfg=self.student.cfg
        )
        if self.train_config.student_cfg.layer_reduction_factor:
            hf, _, _ = load_shallow(
                model_name=self.train_config.student_cfg.model_name,
                mode="hf",
                layer_reduction_factor=self.train_config.student_cfg.layer_reduction_factor,
                selection_method="linspace",
                torch_dtype=self.train_config.torch_dtype,
            )
        else:
            hf = AutoModelForCausalLM.from_pretrained(
                self.train_config.student_cfg.model_name,
                dtype=self.train_config.torch_dtype,
            )
        hf.load_state_dict(student_hf_state_dict)
        hf.save_pretrained(path)

        with open(f"{path}/pl_config.json", "w") as f:
            # cfg = self.train_config.model_dump()
            # json.dump({"train_config": cfg}, f, indent=4)
            json.dump(self.train_config.model_dump(), f, indent=4)

        print(f"Student for task {self.train_config.task.name} saved to {path}")

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
        print(
            f"Student {student_module.train_config.name.upper()} for task {student_module.train_config.task.name} loaded from {path}"
        )
        return student_module
