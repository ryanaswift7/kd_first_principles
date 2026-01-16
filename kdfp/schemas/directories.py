from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from pydantic import computed_field, model_validator

from kdfp.schemas.configBase import ConfigBase
from kdfp.schemas.experiments.experimentConfig import ExperimentConfig
from kdfp.utils.enums import ProjectionTraining


class ExpInstDirs(ConfigBase):
    teacher_model: Path
    student_model: Path
    ray: Path
    lightning: Path
    suite_scores: Path
    cache: Optional[Path] = None
    off_proj_models: Optional[Path] = None

    @computed_field
    @property
    def teacher_scores(self) -> Path:
        return self.teacher_model / "scores"

    @computed_field
    @property
    def student_scores(self) -> Path:
        return self.student_model / "scores"

    @model_validator(mode="after")
    def resolve_paths(self) -> "ExpInstDirs":
        """
        Converts all path fields to absolute, resolved paths.
        Using .resolve() handles symlinks and removes '..' or '.' segments.
        """
        # We can iterate over the fields to avoid repeating 'os.path.abspath' 8 times
        for field_name, value in self.__dict__.items():
            if isinstance(value, Path):
                # .resolve() is the pathlib equivalent of abspath + realpath
                setattr(self, field_name, value.resolve())
            elif value is not None and field_name in self.model_fields:
                # This catches any edge cases where a field might be a string
                setattr(self, field_name, Path(value).resolve())

        return self


@dataclass
class BaseDirs:
    base_cache_dir: Path
    base_model_dir: Path
    ray_dir: Path
    lightning_dir: Path
    root_dir: Path

    def create_instance_dirs(self, exp_cfg: ExperimentConfig) -> "ExpInstDirs":
        """
        Maps base directories to experiment-specific paths using dynamic pair lookup.
        """

        is_offline = exp_cfg.projection.training_method == ProjectionTraining.OFFLINE  # type: ignore
        cache_path = self.base_cache_dir if is_offline else None

        off_proj_path = None
        if exp_cfg.projection.offline_alignment_beta is not None:  # type: ignore
            off_proj_path = (
                self.base_model_dir
                / "offline_projectors"
                / f"off_proj_{exp_cfg.projection.offline_alignment_beta}"  # type: ignore
            )

        exp_model_dir = self.base_model_dir / exp_cfg.suite_title
        exp_model_dir.mkdir(parents=True, exist_ok=True)
        return ExpInstDirs(
            teacher_model=exp_model_dir
            / "teachers"
            / exp_cfg.distill_pair.teacher.name,
            student_model=exp_model_dir / "students" / exp_cfg.name,
            ray=self.ray_dir,
            lightning=self.lightning_dir,
            cache=cache_path,
            off_proj_models=off_proj_path,
            suite_scores=self.root_dir / exp_cfg.suite_title / "scores",
        )


class DirectoryParser(ConfigBase):
    root: str
    cache: str
    models: str
    ray: str
    lightning: str

    def resolve(self) -> BaseDirs:
        """Operates on fields to instantiate a BaseDirs object with absolute Paths."""
        # Convert root to a Path once; resolve() makes it absolute
        root_path = Path(self.root).resolve()

        return BaseDirs(
            base_cache_dir=root_path / self.cache,
            base_model_dir=root_path / self.models,
            ray_dir=root_path / self.ray,
            lightning_dir=root_path / self.lightning,
            root_dir=root_path,
        )

    def get_suite_scores_dir(self, suite_title: str) -> Path:
        """Returns the resolved path for the suite scores directory."""
        root_path = Path(self.root).resolve()
        return root_path / suite_title / "scores"
