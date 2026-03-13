import gc
import json
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
from kdfp.utils.evals import evaluate_model


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

    def test_student(self) -> List[dict]:
        """Class method to evaluate a student model across multiple few-shot settings."""
        print(f"Running tests on student {self.cfg.name}...")

        student_model_path = str(self.dirs.student_model)
        
        master_results_df = pd.DataFrame()
        master_raw_dict = {}
        master_extracted_dict = {}

        for n in self.tasks.num_fewshots:
            print(f"\nBenchmarking all tasks with {n} few-shots...")
            print(f"Tasks: {', '.join(self.tasks.names)}")
            
            extracted_dict, results_df, raw_results_dict = evaluate_model(
                model_path=student_model_path,
                tasks=self.tasks.names,
                experiment_name=self.cfg.name,
                num_fewshot=n,
                batch_size=self.training_params.test_batch_size,
                device="auto",
                limit=self.tasks.sample_limit,
            )
            
            # Nest dictionaries to avoid overwriting across few-shot loops
            fewshot_key = f"fewshot_{n}"
            master_raw_dict[fewshot_key] = raw_results_dict
            master_extracted_dict[fewshot_key] = extracted_dict

            if not results_df.empty:
                if master_results_df.empty:
                    master_results_df = results_df
                else:
                    master_results_df = pd.concat([master_results_df, results_df], ignore_index=True)

        print("\nFinished benchmarking all tasks!")

        if not master_results_df.empty:
            scores_dir = Path(f"{self.dirs.student_model}/scores")
            scores_dir.mkdir(parents=True, exist_ok=True)
            timestamp = get_timestamp()
            
            # Dicts to JSON
            with open(scores_dir / f"raw_eval_scores_{timestamp}.json", "w") as f:
                json.dump(master_raw_dict, f, indent=4)
                
            with open(scores_dir / f"extracted_eval_scores_{timestamp}.json", "w") as f:
                json.dump(master_extracted_dict, f, indent=4)

            # DF to CSV/Parquet
            master_results_df.to_csv(scores_dir / f"results_df_{timestamp}.csv", index=False)
            master_results_df.to_parquet(scores_dir / f"results_df_{timestamp}.parquet", index=False)
            
            print(f"Test scores can be found at: {scores_dir}")
        else:
            print("No valid metrics were found to save.")
            
        self.cleanup_log_dirs() # Assumes this class method exists
        print(f"Testing of student {self.cfg.name} complete!")
        print("=========================================")
        
        return master_results_df.to_dict(orient="records") if not master_results_df.empty else []

