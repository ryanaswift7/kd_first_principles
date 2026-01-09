import os
import shutil
import warnings
from pathlib import Path
from typing import List, Literal, Optional, Union

import lightning.pytorch as pl
import pandas as pd
from pydantic import model_validator

from kdfp.core.hf_tester import HFTesterModule, HFTesterModuleConfig
from kdfp.core.projector import OfflineProjectorModule, OnlineProjector
from kdfp.core.projector.offline.offlineProjector import gen_off_proj_weights_filename
from kdfp.core.projector.randomized import RandomizedProjector
from kdfp.core.student.student import StudentModule
from kdfp.core.teacher import TeacherModule, gen_teacher_weights_filename
from kdfp.schemas.configBase import ConfigBase
from kdfp.schemas.directories import ExpDirConfig
from kdfp.schemas.hparams import HparamsConfig
from kdfp.schemas.lmConfig import LMConfig
from kdfp.schemas.objective_weights import ObjectiveWeights
from kdfp.schemas.projections import ProjectionInstance
from kdfp.schemas.tasks import TaskInstance
from kdfp.schemas.trainingParams import TrainingParamsConfig
from kdfp.utils.activation_caching import activation_caching
from kdfp.utils.enums import ProjectorTraining, StudentReduction
from kdfp.utils.general import get_timestamp
from kdfp.utils.results import generate_result


class ExperimentConfig(ConfigBase):
    name: str
    reduction: StudentReduction
    teacher_cfg: LMConfig
    student_cfg: LMConfig
    dirs: ExpDirConfig
    training_params: TrainingParamsConfig
    lm_params: LMParamsConfig
    projection: Optional[ProjectionInstance]
    objective_weight_variant: str
    objective_weights: ObjectiveWeights
    jsd_beta: Optional[float]
    hparams: HparamsConfig
    tasks: List[TaskInstance]

    @model_validator(mode="after")
    def validate_and_transform_config(self) -> "ExperimentConfig":
        if self.projection:
            if self.reduction == StudentReduction.SHALLOW:
                # all of the fields are already none but I'd prefer
                # to also make the whole object None
                self.projection = None

            elif self.projection.training != ProjectorTraining.OFFLINE:
                self.projection.offline_projector_alignment_beta = None
                self.projection.offline_projector_training_args = None
        return self

    def pretty_print_config(self) -> None:
        """Pretty-prints the complete configuration using Pydantic's JSON serialization."""
        print(self.model_dump_json(indent=4))

    def save_config_json(self, save_dir: Union[str, Path]) -> Path:
        """
        Saves the complete configuration to a 'config.json' file.

        Args:
            save_dir: The directory where the config file will be saved.

        Returns:
            The path to the saved JSON file.
        """
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        file_path = save_path / "config.json"

        # Pydantic's model_dump_json handles the entire serialization process,
        # including nested models, Path objects, enums, etc.
        file_path.write_text(self.model_dump_json(indent=4))

        print(f"Configuration saved to: {file_path}")
        return file_path

    # only offline projectors are initialized here bc don't want to pass in models to
    # lightning modules, so I load them in the student module
    def _init_offline_projector(
        self,
        # cache_dir: str,
        # save_dir: str,
        teacher_dir: Optional[str] = None,
        student_dir: Optional[str] = None,
    ) -> OfflineProjectorModule:
        """Initialize the offline projector for an experiment. Defaults to pretrained teacher
        and student, but you may use a custom version instead by passing in the directories.
        """
        # if we already trained one, load it instead of creating another
        off_proj_weights_path = Path(
            self.dirs.off_proj_models
        ) / gen_off_proj_weights_filename(
            alignment_beta=self.projection.offline_projector_alignment_beta
        )
        if off_proj_weights_path.exists():
            print(
                f"Pretrained offline projector for with alignment beta '{self.projection.offline_projector_alignment_beta}' located at: {str(off_proj_weights_path)}"
            )

            # p = OfflineProjectorModule.load_pretrained(self.dirs.off_proj_model)
            # return p.extract_single_direction(direction=self.projection.direction)
        else:
            # cache teacher and student acts
            teacher_cache_dir, teacher_dim = activation_caching(
                lm_name=self.teacher_cfg.model_name,
                lm_hook_points=self.teacher_cfg.hook_list,
                dataset_name=self.projection.offline_projector_training_args.dataset.name,
                dataset_subset=self.projection.offline_projector_training_args.dataset.subset,
                split=self.projection.offline_projector_training_args.dataset.split,
                seq_len=self.training_params.seq_len,
                batch_size=self.projection.offline_projector_training_args.cache_batch_size,
                cache_dir=self.dirs.cache,
                token_rename=self.projection.offline_projector_training_args.dataset.token_rename,
                shuffle=self.projection.offline_projector_training_args.dataset.shuffle,
                lm_dir=teacher_dir,
                total_tokens=self.projection.offline_projector_training_args.total_tokens,
                force_cache_all=self.projection.offline_projector_training_args.force_cache_all,
                cache_outputs=False,
                num_workers=self.training_params.resources.projector.num_data_workers,
                device=self.training_params.device,
                torch_dtype=self.training_params.torch_dtype,
            )

            student_cache_dir, student_dim = activation_caching(
                lm_name=self.student_cfg.model_name,
                lm_hook_points=self.student_cfg.hook_list,
                dataset_name=self.projection.offline_projector_training_args.dataset.name,
                dataset_subset=self.projection.offline_projector_training_args.dataset.subset,
                split=self.projection.offline_projector_training_args.dataset.split,
                seq_len=self.training_params.seq_len,
                batch_size=self.projection.offline_projector_training_args.cache_batch_size,
                cache_dir=self.dirs.cache,
                token_rename=self.projection.offline_projector_training_args.dataset.token_rename,
                shuffle=self.projection.offline_projector_training_args.dataset.shuffle,
                lm_dir=student_dir,
                total_tokens=self.projection.offline_projector_training_args.total_tokens,
                force_cache_all=self.projection.offline_projector_training_args.force_cache_all,
                cache_outputs=False,
                num_workers=self.training_params.resources.projector.num_data_workers,
                device=self.training_params.device,
                torch_dtype=self.training_params.torch_dtype,
            )

            # create projector
            projector = OfflineProjectorModule.create(
                teacher_cache_dir=teacher_cache_dir,
                student_cache_dir=student_cache_dir,
                teacher_dim=teacher_dim,
                student_dim=student_dim,
                projection=self.projection,
                hyperparams=self.hparams,
                training_params=self.training_params,
                dirs=self.dirs,
                val_split=0.1,
                test_split=0.1,
            )
            projector.save_pretrained(path=self.dirs.off_proj_models)
            # load_test = OfflineProjector.load_pretrained(save_path=self.dirs.off_proj_models)
            # return projector.extract_single_direction(direction=self.projection.direction)

    def init_online_projector(self):
        return OnlineProjector(
            teacher_dim=self.teacher_cfg.d_model,
            student_dim=self.student_cfg.d_model,
            direction=self.projection.direction,
            torch_dtype=self.torch_dtype,
        )

    def init_randomized_projector(self):
        return RandomizedProjector(
            teacher_dim=self.teacher_cfg.d_model,
            student_dim=self.student_cfg.d_model,
            direction=self.projection.direction,
            torch_dtype=self.torch_dtype,
        )

    def init_projector(self) -> None:
        """Initialize projector for an experiment. kwargs are for init_offline_projector."""
        if self.projection:
            if self.projection.training == ProjectorTraining.ONLINE:
                # self.init_online_projector()
                print(
                    f"{self.projection.training.value.upper()} projector will be initalized during the experiment."
                )

            elif self.projection.training == ProjectorTraining.OFFLINE:
                print(
                    f"{self.projection.training.value.upper()} projector initializtion starting..."
                )
                self._init_offline_projector()
                print(
                    f"{self.projection.training.value.upper()} projector initializtion complete!"
                )

            elif self.projection.training == ProjectorTraining.RANDOMIZED:
                # self.init_randomized_projector()
                print(
                    f"{self.projection.training.value.upper()} projector will be initalized during the experiment."
                )

            else:
                raise ValueError(
                    f"Invalid projection training specified: {self.projection.training}."
                )

        return

    def init_teacher_by_task(self, task: TaskInstance) -> None:
        # save_dir = f"{self.dirs.teacher_model}/{task.task_name}"
        teacher_weights_file = (
            Path(self.dirs.teacher_model)
            / task.name
            / gen_teacher_weights_filename(task.name)
        )
        if teacher_weights_file.exists():
            print(
                f"Pretrained teacher for task '{task.name}' located at: {str(teacher_weights_file)}"
            )
        else:
            t = TeacherModule.create(
                lm_name=self.teacher_cfg.model_name,
                task=task,
                lm_params=self.lm_params,
                hyperparams=self.hparams,
                training_params=self.training_params,
                dirs=self.dirs,
            )
            t.save_pretrained(path=f"{self.dirs.teacher_model}/{task.name}")

    def init(self) -> None:
        """Initialize teachers, student, and projector"""
        self.init_projector()
        for task in self.tasks:
            self.init_teacher_by_task(task)

    def cleanup_log_dirs(self) -> None:
        shutil.rmtree(self.dirs.ray, ignore_errors=True)
        shutil.rmtree(self.dirs.lightning, ignore_errors=True)

    def test_trainer_factory(self) -> pl.Trainer:
        return pl.Trainer(
            accelerator="gpu",
            strategy="auto",
            devices=1,
            deterministic=self.training_params.reproducible_backend,
            default_root_dir=self.dirs.lightning,
        )

    def run(self) -> List[dict]:
        results_list = []

        for i, task in enumerate(self.tasks):
            print(f"\nRunning task {i + 1}/{len(self.tasks)}: '{task.name.upper()}'...")

            # DO NOT OVERWRITE EXISTING MODELS
            model_save_dir = f"{self.dirs.student_model}/{task.name}"
            if os.path.isdir(model_save_dir) and os.listdir(model_save_dir):
                warnings.warn(
                    "Student model directory exists and is not empty!\n"
                    + f"Skipping experiment {self.name.upper()} to avoid overwriting models..."
                )
                continue

            student_module = StudentModule.create(
                exp_name=self.name,
                dirs=self.dirs,
                student_cfg=self.student_cfg,
                teacher_cfg=self.teacher_cfg,
                task=task,
                lm_params=self.lm_params,
                hyperparams=self.hparams,
                training_params=self.training_params,
                projection=self.projection,
                jsd_beta=self.jsd_beta,
                objective_weights=self.objective_weights,
            )
            student_module.save_pretrained(model_save_dir)
            results = self.test_trainer_factory().test(
                model=student_module, verbose=True
            )
            new_result = generate_result(
                name=self.name,
                model=student_module.student,
                task_name=task.name,
                metric_name="accuracy",
                score=results[0]["test_acc"],
            )
            results_list.append(new_result)
            new_result_df = pd.DataFrame([new_result])

            scores_dir = Path(f"{self.dirs.student_scores}/{task.name}")
            scores_dir.mkdir(parents=True, exist_ok=True)
            new_result_df.to_parquet(
                scores_dir / f"results_df_{get_timestamp()}.parquet"
            )

            print(f"\nFinished task {i + 1}/{len(self.tasks)}: '{task.name.upper()}'!")
        self.cleanup_log_dirs()
        return results_list

    def test_student(self, mode: Literal["pl", "hf"]) -> List[dict]:
        print(f"Running tests on student {self.name}...")
        results_list = []

        for i, task in enumerate(self.tasks):
            print(f"\nBenchmarking student {self.name} on task {task.name.upper()}...")
            student_save_dir = f"{self.dirs.student_model}/{task.name}"
            if mode == "pl":
                student_module = StudentModule.load_pretrained(student_save_dir)
            else:
                student_test_cfg = HFTesterModuleConfig(
                    model_path=student_save_dir,
                    tokenizer_name=self.student_cfg.model_name,
                    task=task,
                    seq_len=self.training_params.seq_len,
                    batch_size=self.lm_params.test_batch_size,
                    seed=self.training_params.seed,
                    torch_dtype=self.training_params.torch_dtype,
                    num_data_workers=self.training_params.resources.student.num_data_workers,
                    n_samples=self.lm_params.num_samples,
                )
                student_module = HFTesterModule(student_test_cfg)
            results = self.test_trainer_factory().test(
                model=student_module, verbose=True
            )
            new_result = generate_result(
                name=self.name,
                model=student_module.student,
                task_name=task.name,
                metric_name="accuracy",
                score=results[0]["test_acc"],
            )
            results_list.append(new_result)
            new_result_df = pd.DataFrame([new_result])

            scores_dir = Path(f"{self.dirs.student_scores}/{task.name}")
            scores_dir.mkdir(parents=True, exist_ok=True)
            new_result_df.to_parquet(
                scores_dir / f"results_df_{get_timestamp()}.parquet"
            )

            print(f"Finished with task {task.name.upper()}!")
        self.cleanup_log_dirs()
        return results_list

    def test_teachers(self, mode: Literal["pl", "hf"]) -> List[dict]:
        print("Running tests on teachers...")
        results_list = []

        for i, task in enumerate(self.tasks):
            print(f"\nBenchmarking teacher finetuned on task {task.name.upper()}...")
            teacher_save_dir = f"{self.dirs.teacher_model}/{task.name}"

            if mode == "pl":
                teacher_module = TeacherModule.load_pretrained(teacher_save_dir)
                # adjust current n_samples to respect new config (used for quick debugging)
                teacher_module.train_config.n_samples = self.lm_params.num_samples
            else:
                teacher_test_cfg = HFTesterModuleConfig(
                    model_path=teacher_save_dir,
                    tokenizer_name=self.teacher_cfg.model_name,
                    task=task,
                    seq_len=self.training_params.seq_len,
                    batch_size=self.lm_params.test_batch_size,
                    seed=self.training_params.seed,
                    torch_dtype=self.training_params.torch_dtype,
                    num_data_workers=self.training_params.resources.teacher.num_data_workers,
                    n_samples=self.lm_params.num_samples,
                )
                teacher_module = HFTesterModule(teacher_test_cfg)

            results = self.test_trainer_factory().test(
                model=teacher_module, verbose=True
            )
            new_result = generate_result(
                name="teacher_ft",
                model=teacher_module.model,
                task_name=task.name,
                metric_name="accuracy",
                score=results[0]["test_acc"],
            )
            results_list.append(new_result)
            new_result_df = pd.DataFrame([new_result])

            scores_dir = Path(f"{self.dirs.teacher_scores}/{task.name}")
            scores_dir.mkdir(parents=True, exist_ok=True)
            new_result_df.to_parquet(
                scores_dir / f"results_df_{get_timestamp()}.parquet"
            )

            print(f"Finished with task {task.name.upper()}!")
        self.cleanup_log_dirs()
        return results_list
