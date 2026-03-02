import gc
import os
import shutil
import warnings
from pathlib import Path
from typing import List, Optional

import lightning.pytorch as pl
import lm_eval
import pandas as pd
import torch

from kdfp.core.projector import OfflineProjectorModule
from kdfp.core.projector.offline.offlineProjector import gen_off_proj_weights_filename
from kdfp.core.student.student import StudentModule
from kdfp.core.teacher import TeacherModule, gen_teacher_weights_filename
from kdfp.schemas.configBase import ConfigBase
from kdfp.schemas.directories import ExpInstDirs
from kdfp.schemas.experiments.experimentConfig import ExperimentConfig
from kdfp.schemas.hparams import HparamsConfig
from kdfp.schemas.tasks import LMEvalTasks
from kdfp.schemas.trainingParams import TrainingParams
from kdfp.utils.activation_caching import (
    cache_student_and_teacher_acts,
)
from kdfp.utils.enums import ProjectionTraining
from kdfp.utils.general import get_timestamp
from kdfp.utils.evals import generate_result


class ExperimentInstance(ConfigBase):
    cfg: ExperimentConfig
    dirs: ExpInstDirs
    training_params: TrainingParams
    hparams: HparamsConfig
    tasks: LMEvalTasks

    # only offline projectors are initialized here bc don't want to pass in models to
    # lightning modules, so I load them in the student module
    def _init_offline_projector(
        self,
        # cache_dir: str,
        # save_dir: str,
        local_teacher_dir: Optional[str] = None,
        local_student_dir: Optional[str] = None,
    ) -> None:
        """Initialize the offline projector for an experiment. Defaults to pretrained teacher
        and student, but you may use custom local versions instead by passing in the directories.
        """
        # if we already trained one, load it instead of creating another
        off_proj_weights_path = Path(
            self.dirs.off_proj_models  # type: ignore
        ) / gen_off_proj_weights_filename(
            alignment_beta=self.cfg.projection.offline_alignment_beta  # type: ignore
        )
        if off_proj_weights_path.exists():
            print(
                f"Pretrained offline projector for with alignment beta '{self.cfg.projection.offline_alignment_beta}' located at: {str(off_proj_weights_path)}"  # type: ignore
            )

            # p = OfflineProjectorModule.load_pretrained(self.dirs.off_proj_model)
            # return p.extract_single_direction(direction=self.projection.direction)
        else:
            # cache dir should always be specified for offline projector
            if cache_dir := self.dirs.cache:
                (teacher_cache_dir, teacher_dim, student_cache_dir, student_dim) = (
                    cache_student_and_teacher_acts(
                        cfg=self.cfg,
                        training_params=self.training_params,
                        cache_dir=str(cache_dir),
                        cache_outputs=False,
                        local_teacher_dir=local_teacher_dir,
                        local_student_dir=local_student_dir,
                    )
                )

            # create projector
            projector = OfflineProjectorModule.create(
                teacher_cache_dir=teacher_cache_dir,
                student_cache_dir=student_cache_dir,
                teacher_dim=teacher_dim,
                student_dim=student_dim,
                projection=self.cfg.projection,  # type: ignore
                hyperparams=self.hparams,
                training_params=self.training_params,
                dirs=self.dirs,
                val_split=0.1,
                test_split=0.1,
                seed=self.cfg.seed,
            )
            projector.save_pretrained(path=self.dirs.off_proj_models)  # type: ignore
            # load_test = OfflineProjector.load_pretrained(save_path=self.dirs.off_proj_models)
            # return projector.extract_single_direction(direction=self.projection.direction)

    def init_projector(self) -> None:
        """Initialize projector for an experiment. kwargs are for init_offline_projector."""
        if proj := self.cfg.projection:
            if proj.training_method == ProjectionTraining.ONLINE:
                print(
                    f"{proj.training_method.value.upper()} projector will be initalized during the experiment."
                )

            elif proj.training_method == ProjectionTraining.OFFLINE:
                print(
                    f"{proj.training_method.value.upper()} projector initializtion starting..."
                )
                self._init_offline_projector()
                print(
                    f"{proj.training_method.value.upper()} projector initializtion complete!"
                )

            elif proj.training_method == ProjectionTraining.RANDOMIZED:
                print(
                    f"{proj.training_method.value.upper()} projector will be initalized during the experiment."
                )

            else:
                raise ValueError(
                    f"Invalid projection training specified: {proj.training_method}."
                )

        return

    def init_ft_teacher(self) -> None:
        teacher_weights_file = (
            Path(self.dirs.teacher_model) / gen_teacher_weights_filename()
        )
        if teacher_weights_file.exists():
            print(f"Pretrained teacher located at: {str(teacher_weights_file)}")
        else:
            t = TeacherModule.create(
                lm_name=self.cfg.distill_pair.teacher.name,
                hyperparams=self.hparams,
                training_params=self.training_params,
                dirs=self.dirs,
                seed=self.cfg.seed,
            )
            t.save_pretrained(path=f"{self.dirs.teacher_model}/{t.model_file}")

    def init(self) -> None:
        """Initialize finetuned teachers and projectors (if necessary)"""
        if self.cfg.projection:
            self.init_projector()
        if self.training_params.controls.finetune_teacher:
            self.init_ft_teacher()

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
        # DO NOT OVERWRITE EXISTING MODELS
        student_already_exists = False
        model_save_dir = str(self.dirs.student_model)
        if os.path.isdir(model_save_dir) and os.listdir(model_save_dir):
            warnings.warn(
                "Student model directory exists and is not empty!\n"
                + f"Skipping experiment {self.cfg.name.upper()} to avoid overwriting models..."
            )
            student_already_exists = True

        if not student_already_exists:
            print(f"Training student {self.cfg.name}'...")
            student_module = StudentModule.create(
                cfg=self.cfg,
                dirs=self.dirs,
                hyperparams=self.hparams,
                training_params=self.training_params,
            )
            student_module.save_pretrained(model_save_dir)
            print(f"Student {self.cfg.name} training complete!")

        return self.test_student()

        # results = self.test_trainer_factory().test(model=student_module, verbose=True)

        # new_result = generate_result(
        #     name=self.cfg.name,
        #     model=student_module.student,
        #     task_name=self.training_params.dataset.name.split("/")[-1],
        #     metric_name="perplexity",
        #     score=results[0]["test_ppl"],
        # )
        # results_list.append(new_result)

        # for i, task_name in enumerate(self.tasks.names):
        #     print(
        #         f"\nBenchmarking task {i + 1}/{len(self.tasks.names)}: '{task_name.upper()}'..."
        #     )
        #     for n in self.tasks.num_fewshots:
        #         print(f"  - with {n} few-shots...")
        #         task_result = lm_eval.simple_evaluate(
        #             model="hf",
        #             model_args=f"pretrained={model_save_dir},dtype=auto",
        #             tasks=[task_name],
        #             num_fewshot=n,
        #             batch_size="auto",
        #             device="cuda:0",
        #         )

        #         new_result = generate_result(
        #             name=self.cfg.name,
        #             model=student_module.student,
        #             task_name=task_name,
        #             metric_name="accuracy",
        #             num_fewshot=n,
        #             score=task_result["results"][task_name]["acc,none"],
        #         )
        #         results_list.append(new_result)
        #     print(
        #         f"\nFinished task {i + 1}/{len(self.tasks.names)}: '{task_name.upper()}'!"
        #     )

        # results_df = pd.DataFrame(results_list)
        # scores_dir = Path(f"{self.dirs.student_model}/scores")
        # scores_dir.mkdir(parents=True, exist_ok=True)
        # results_df.to_markdown(scores_dir / f"results_df_{get_timestamp()}.md")
        # self.cleanup_log_dirs()
        # return results_list

    # def test_student(self, mode: Literal["pl", "hf"]) -> List[dict]:
    #     print(f"Running tests on student {self.cfg.name}...")
    #     results_list = []

    #     student_module = AutoModelForCausalLM.from_pretrained(self.dirs.student_model)
    #     student_module.eval()

    #     for i, task_name in enumerate(self.tasks.names):
    #         print(
    #             f"\nBenchmarking student {self.cfg.name} on task {task_name.upper()}..."
    #         )
    #         student_save_dir = f"{self.dirs.student_model}/{task_name}"
    #         if mode == "pl":
    #             student_module = StudentModule.load_pretrained(student_save_dir)
    #         else:
    #             student_test_cfg = HFTesterModuleConfig(
    #                 model_path=student_save_dir,
    #                 tokenizer_name=self.cfg.distill_pair.student.name,
    #                 task=task_name,
    #                 seq_len=self.training_params.seq_len,
    #                 batch_size=self.training_params.test_batch_size,
    #                 seed=self.cfg.seed,
    #                 torch_dtype=self.training_params.torch_dtype,
    #                 num_data_workers=self.training_params.resources.student.num_data_workers,
    #                 n_samples=self.training_params.num_samples,
    #             )
    #             student_module = HFTesterModule(student_test_cfg)
    #         results = self.test_trainer_factory().test(
    #             model=student_module, verbose=True
    #         )
    #         new_result = generate_result(
    #             name=self.cfg.name,
    #             model=student_module.student,
    #             task_name=task_name,
    #             metric_name="accuracy",
    #             score=results[0]["test_acc"],
    #         )
    #         results_list.append(new_result)
    #         new_result_df = pd.DataFrame([new_result])

    #         scores_dir = Path(f"{self.dirs.student_model}/scores")
    #         scores_dir.mkdir(parents=True, exist_ok=True)
    #         new_result_df.to_markdown(scores_dir / f"results_df_{get_timestamp()}.md")

    #         print(f"Finished with task {task_name.upper()}!")
    #     self.cleanup_log_dirs()
    #     return results_list

    def test_student(self) -> List[dict]:
        print(f"Running tests on student {self.cfg.name}...")
        results_list = []

        student_module = StudentModule.load_pretrained(str(self.dirs.student_model))


        for i, task_name in enumerate(self.tasks.names):
            print(
                f"\nBenchmarking task {i + 1}/{len(self.tasks.names)}: '{task_name.upper()}'..."
            )
            for n in self.tasks.num_fewshots:
                print(f"  - with {n} few-shots...")
                task_result = lm_eval.simple_evaluate(
                    model="hf",
                    model_args=f"pretrained={self.dirs.student_model},dtype=auto",
                    # include_path="kdfp/tasks"
                    tasks=[task_name],
                    num_fewshot=n,
                    batch_size=self.training_params.test_batch_size,
                    device="cuda",
                    # limit=1,
                )

                new_result = generate_result(
                    name=self.cfg.name,
                    model=student_module.student,
                    task_name=task_name,
                    metric_name="accuracy",
                    num_fewshot=n,
                    score=task_result["results"][task_name]["acc,none"],
                )
                results_list.append(new_result)
            print(
                f"\nFinished task {i + 1}/{len(self.tasks.names)}: '{task_name.upper()}'!"
            )

        results_df = pd.DataFrame(results_list)
        scores_dir = Path(f"{self.dirs.student_model}/scores")
        scores_dir.mkdir(parents=True, exist_ok=True)
        results_df.to_markdown(scores_dir / f"results_df_{get_timestamp()}.md")
        self.cleanup_log_dirs()
        print(f"Testing of student {self.cfg.name} complete!")
        print(f"Test scores can be found at: {scores_dir}")
        print("=========================================")
        torch.cuda.empty_cache()
        gc.collect()
        return results_list

    # def test_teachers(self, mode: Literal["pl", "hf"]) -> List[dict]:
    #     pl.seed_everything(seed=self.cfg.seed, workers=True)
    #     results_list = []
    #     base_teacher_results_list = []

    #     base_teacher_scores_dir = self.dirs.suite_scores / "base_teacher"
    #     base_teacher_scores_dir.mkdir(parents=True, exist_ok=True)

    #     for i, task in enumerate(self.task_list):
    #         print(f"Benchmarking baseline teacher on task {task.name.upper()}...")
    #         base_teacher_cfg = BaselineModuleConfig(
    #             model_name=self.cfg.distill_pair.teacher.name,
    #             is_shallow=False,
    #             task=task,
    #             seq_len=self.training_params.seq_len,
    #             batch_size=self.training_params.test_batch_size,
    #             seed=self.cfg.seed,
    #             torch_dtype=self.training_params.torch_dtype,
    #             num_data_workers=self.training_params.resources.teacher.num_data_workers,
    #             n_samples=self.training_params.num_samples,
    #             layer_reduction_factor=None,
    #             stream_data=self.training_params.resources.stream_task_data,
    #         )
    #         base_teacher_module = BaselineModule(base_teacher_cfg)
    #         base_teacher_module.print_param_count("base_teacher")
    #         base_teacher_results = self.test_trainer_factory().test(
    #             model=base_teacher_module, verbose=True
    #         )
    #         base_teacher_results_list.append(
    #             generate_result(
    #                 name="base_teacher",
    #                 model=base_teacher_module.model,
    #                 task_name=task.name,
    #                 metric_name="accuracy",
    #                 score=base_teacher_results[0]["test_acc"],
    #             )
    #         )

    #         if self.training_params.controls.finetune_teacher:
    #             print(
    #                 f"\nBenchmarking teacher finetuned on task {task.name.upper()}..."
    #             )
    #             teacher_save_dir = f"{self.dirs.teacher_model}/{task.name}"

    #             if mode == "pl":
    #                 teacher_module = TeacherModule.load_pretrained(teacher_save_dir)
    #                 # adjust current n_samples to respect new config (used for quick debugging)
    #                 teacher_module.train_config.n_samples = (
    #                     self.training_params.num_samples
    #                 )
    #             else:
    #                 teacher_test_cfg = HFTesterModuleConfig(
    #                     model_path=teacher_save_dir,
    #                     tokenizer_name=self.cfg.distill_pair.teacher.name,
    #                     task=task,
    #                     seq_len=self.training_params.seq_len,
    #                     batch_size=self.training_params.test_batch_size,
    #                     seed=self.cfg.seed,
    #                     torch_dtype=self.training_params.torch_dtype,
    #                     num_data_workers=self.training_params.resources.teacher.num_data_workers,
    #                     n_samples=self.training_params.num_samples,
    #                 )
    #                 teacher_module = HFTesterModule(teacher_test_cfg)

    #             results = self.test_trainer_factory().test(
    #                 model=teacher_module, verbose=True
    #             )
    #             new_result = generate_result(
    #                 name="teacher_ft",
    #                 model=teacher_module.model,
    #                 task_name=task.name,
    #                 metric_name="accuracy",
    #                 score=results[0]["test_acc"],
    #             )
    #             results_list.append(new_result)
    #             new_result_df = pd.DataFrame([new_result])

    #             scores_dir = Path(f"{self.dirs.teacher_model}/{task.name}/scores")
    #             scores_dir.mkdir(parents=True, exist_ok=True)
    #             new_result_df.to_parquet(
    #                 scores_dir / f"results_df_{get_timestamp()}.parquet"
    #             )
    #             new_result_df.to_markdown(
    #                 scores_dir / f"results_df_{get_timestamp()}.md"
    #             )
    #             print(f"Finished with task {task.name.upper()}!")

    #     base_teacher_results_df = pd.DataFrame(base_teacher_results_list)
    #     base_teacher_results_df.to_parquet(
    #         base_teacher_scores_dir / f"results_df_{get_timestamp()}.parquet"
    #     )
    #     base_teacher_results_df.to_markdown(
    #         base_teacher_scores_dir / f"results_df_{get_timestamp()}.md"
    #     )
    #     self.cleanup_log_dirs()
    #     return results_list
