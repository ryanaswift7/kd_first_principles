import itertools
from typing import List

from pydantic import computed_field

from kdfp.schemas.configBase import ConfigBase
from kdfp.schemas.experiments.experimentConfig import (
    ExperimentConfig,
    ObjectiveInstance,
)
from kdfp.schemas.experiments.experimentFilter import ExperimentFilter
from kdfp.schemas.models import DistillationGroups
from kdfp.schemas.projections import ProjectionInstance, ProjectionParser
from kdfp.utils.enums import (
    ObjectiveWeightStr,
    ProjectionTraining,
    StudentType,
    TrainingType,
)


class ObjectiveParams(ConfigBase):
    weight_variants: List[ObjectiveWeightStr]
    jsd_betas: List[float]


class ExperimentSpace(ConfigBase):
    seeds: List[int]
    student_types: List[StudentType]
    objective: ObjectiveParams
    projection: ProjectionParser
    filter: ExperimentFilter

    @computed_field
    @property
    def default_seed(self) -> int:
        return self.seeds[0]

    def generate_experiment_cfgs(
        self, suite_title: str, distill_groups: DistillationGroups, get_baselines: bool
    ) -> List[ExperimentConfig]:
        kd_list = self.generate_kd_cfgs(
            suite_title=suite_title, distill_groups=distill_groups
        )
        baseline_list = (
            self.generate_baseline_cfgs(
                suite_title=suite_title, distill_groups=distill_groups
            )
            if get_baselines
            else []
        )
        return self._apply_filter(kd_list + baseline_list)

    def generate_kd_cfgs(
        self, suite_title: str, distill_groups: DistillationGroups
    ) -> List[ExperimentConfig]:
        """
        Generates, names, and filters the suite of experiments.
        """
        candidates: List[ExperimentConfig] = []

        for seed, student in itertools.product(self.seeds, self.student_types):
            if student == StudentType.SHALLOW:
                for obj, jsd in itertools.product(
                    self.objective.weight_variants, self.objective.jsd_betas
                ):
                    candidates.append(
                        ExperimentConfig(
                            suite_title=suite_title,
                            seed=seed,
                            student_type=student,
                            distill_pair=getattr(distill_groups, student.value),
                            objective=ObjectiveInstance(
                                weight_str=obj,
                                jsd_beta=jsd,
                            ),
                            training_type=TrainingType.KD,
                        )
                    )

            else:
                for direction, training in itertools.product(
                    self.projection.directions, self.projection.training_methods
                ):
                    betas = (
                        self.projection.offline_alignment_betas
                        if training == ProjectionTraining.OFFLINE
                        else [None]
                    )

                    for beta, obj, jsd in itertools.product(
                        betas, self.objective.weight_variants, self.objective.jsd_betas
                    ):
                        candidates.append(
                            ExperimentConfig(
                                suite_title=suite_title,
                                seed=seed,
                                student_type=student,
                                distill_pair=getattr(DistillationGroups, student.value),
                                projection=ProjectionInstance(
                                    direction=direction,
                                    training_method=training,
                                    offline_alignment_beta=beta,
                                ),
                                objective=ObjectiveInstance(
                                    weight_str=obj,
                                    jsd_beta=jsd,
                                ),
                                training_type=TrainingType.KD,
                            )
                        )

        return candidates

    def generate_baseline_cfgs(
        self, suite_title: str, distill_groups: DistillationGroups
    ) -> List[ExperimentConfig]:
        """
        Generates experiments to test all selected student types with finetuning and with no training,
        as well as testing the teacher model.
        """
        candidates: List[ExperimentConfig] = []

        # first make finetuning experiments
        for seed, student in itertools.product(self.seeds, self.student_types):
            candidates.append(
                ExperimentConfig(
                    suite_title=suite_title,
                    seed=seed,
                    student_type=student,
                    distill_pair=getattr(distill_groups, student.value),
                    objective=ObjectiveInstance(
                        weight_str=ObjectiveWeightStr.FINETUNING,
                        jsd_beta=None,
                    ),
                    training_type=TrainingType.FT,
                )
            )

        # then make no train (using default seed)
        for student in self.student_types:
            candidates.append(
                ExperimentConfig(
                    suite_title=suite_title,
                    seed=self.default_seed,
                    student_type=student,
                    distill_pair=getattr(distill_groups, student.value),
                    objective=ObjectiveInstance(
                        weight_str=ObjectiveWeightStr.NO_TRAIN,
                        jsd_beta=-1,
                    ),
                    training_type=TrainingType.NO_TRAIN,
                )
            )

        return candidates

    def _apply_filter(
        self, experiments: List[ExperimentConfig]
    ) -> List[ExperimentConfig]:
        """Filters experiments based on name matching."""
        filtered = []

        include_list = self.filter.include or []
        exclude_list = self.filter.exclude or []

        for exp in experiments:
            name = exp.name

            # Handle Inclusion: If list exists, name must match at least one entry
            if include_list:
                if not any(match in name for match in include_list):
                    continue

            # Handle Exclusion: If list exists, name must NOT match any entry
            if exclude_list:
                if any(match in name for match in exclude_list):
                    continue

            filtered.append(exp)

        return filtered
