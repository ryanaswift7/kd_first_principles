from kdfp.utils.activation_caching import activation_caching
from kdfp.core.projector import OfflineProjector, OnlineProjector, OFF_PROJ_STATE_DICT_FILE
from typing import Optional, List
import torch
import torch.nn as nn
from kdfp.schemas.lm import LM
from kdfp.schemas.subConfigs import HyperparamsConfig, ExpProjectionConfig, OffProjTrainArgsConfig, TrainingParamsConfig
from pathlib import Path


# initialize (finetuned) teacher
# - have a folder with the teacher name
# - subdirs for each task
# - each subdir has one model (i.e. one model per task)

"""
initialize offline projector
- probably need to grab caching script and cache some acts
- only need to cache for teacher and narrow student (i.e. 410m)
"""

def _init_offline_projector(teacher: LM,
                            student: LM,
                            cache_dir: str,
                            save_dir: str,
                            projection: ExpProjectionConfig,
                            off_proj_train_args: OffProjTrainArgsConfig,
                            train_params: TrainingParamsConfig,
                            hyperparams: HyperparamsConfig,
                            teacher_dir: Optional[str] = None,
                            student_dir: Optional[str] = None,) -> OfflineProjector:
    # if we already trained one, load it instead of creating another
    weights_path = Path(save_dir) / OFF_PROJ_STATE_DICT_FILE
    if weights_path.exists():
        p = OfflineProjector.load_pretrained(str(weights_path))
        return p.extract_single_direction(direction=projection.direction)

    # cache teacher and student acts
    teacher_cache_dir, teacher_dim = activation_caching(lm_name=teacher.model_name,
                                               lm_hook_points=teacher.hook_list,
                                               dataset_name=off_proj_train_args.dataset.name,
                                               dataset_subset=off_proj_train_args.dataset.subset,
                                               split=off_proj_train_args.dataset.split,
                                               seq_len=train_params.seq_len,
                                               batch_size=off_proj_train_args.cache_batch_size,
                                               cache_dir=cache_dir,
                                               token_rename=off_proj_train_args.dataset.token_rename,
                                               shuffle=off_proj_train_args.dataset.shuffle,
                                               lm_dir=teacher_dir,
                                               total_tokens=off_proj_train_args.total_tokens,
                                               force_cache_all=off_proj_train_args.force_cache_all,
                                               cache_outputs=False,
                                               num_workers=train_params.num_data_workers,
                                               device=train_params.device)
    
    student_cache_dir, student_dim = activation_caching(lm_name=student.model_name,
                                               lm_hook_points=student.hook_list,
                                               dataset_name=off_proj_train_args.dataset.name,
                                               dataset_subset=off_proj_train_args.dataset.subset,
                                               split=off_proj_train_args.dataset.split,
                                               seq_len=train_params.seq_len,
                                               batch_size=off_proj_train_args.cache_batch_size,
                                               cache_dir=cache_dir,
                                               token_rename=off_proj_train_args.dataset.token_rename,
                                               shuffle=off_proj_train_args.dataset.shuffle,
                                               lm_dir=student_dir,
                                               total_tokens=off_proj_train_args.total_tokens,
                                               force_cache_all=off_proj_train_args.force_cache_all,
                                               cache_outputs=False,
                                               num_workers=train_params.num_data_workers,
                                               device=train_params.device)

    # create projector
    projector = OfflineProjector.create(
        teacher_cache_dir=teacher_cache_dir,
        student_cache_dir=student_cache_dir,
        teacher_dim=teacher_dim,
        student_dim=student_dim,
        alignment_beta=projection.offline_projector_alignment_beta,
        max_epochs=train_params.num_epochs,
        weight_decay_choices=hyperparams.weight_decay,
        max_grad_norm_choices=hyperparams.max_grad_norm,
        batch_sizes=hyperparams.batch_size,
        n_trials=off_proj_train_args.n_trials,
        save_dir=f"{save_dir}/projector_{projection.offline_projector_alignment_beta}",
        seed=train_params.seed,
        reproducible_backend=train_params.reproducible_backend,
        num_gpus_per_trial=off_proj_train_args.num_gpus_per_trial,
        num_cpus_per_trial=off_proj_train_args.num_cpus_per_trial,
    )
    projector.save_pretrained(save_path=save_dir)
    return projector.extract_single_direction(direction=projection.direction)

def _init_online_projector(teacher: LM, student: LM, projection: ExpProjectionConfig):
    return OnlineProjector(teacher_dim=teacher.d_model,
                           student_dim=student.d_model,
                           direction=projection.direction)

def init_projector(teacher: LM,
                   student: LM, 
                   projection: ExpProjectionConfig,
                   **kwargs) -> nn.Module:
    """Initialize projector for an experiment. kwargs are for init_offline_projector."""
    if projection.training == 'online':
        return _init_online_projector(teacher=teacher, student=student, projection=projection)
    elif projection.training == 'offline':
        return _init_offline_projector(teacher=teacher, student=student, projection=projection, **kwargs)
    else:
        raise ValueError(f"Invalid projection training specified: {projection.training}. Must be either 'online' or 'offline'.")