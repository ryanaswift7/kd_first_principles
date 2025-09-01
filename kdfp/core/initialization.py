from kdfp.utils.activation_caching import activation_caching
from kdfp.core.projector import Projector
from typing import Optional, List
import torch
from kdfp.schemas.lm import LM

# initialize (finetuned) teacher
# - have a folder with the teacher name
# - subdirs for each task
# - each subdir has one model (i.e. one model per task)

"""
initialize offline projector
- probably need to grab caching script and cache some acts
- only need to cache for teacher and narrow student (i.e. 410m)
"""

def init_projector(teacher: LM,
                   student: LM,
                   alignment_beta: float,
                   cache_dir: str,
                   save_dir: str,
                   seq_len: int,
                   cache_batch_size: int,
                   num_epochs: int,
                   weight_decay_choices: List[float],
                   max_grad_norm_choices: List[float],
                   batch_size_choices: List[int],
                   dataset_name: str,
                   split: str,
                   dataset_subset: Optional[str] = None,
                   token_rename: bool = False,
                   shuffle: bool = False,
                   n_trials: int = 15,
                   teacher_dir: Optional[str] = None,
                   student_dir: Optional[str] = None,
                   total_tokens: Optional[int] = None,
                   num_workers: Optional[int] = None,
                   force_cache_all: bool = False,
                   device: str = "cuda" if torch.cuda.is_available() else 'cpu',
                   seed: int = 44) -> Projector:
    # cache teacher and student acts
    teacher_cache_dir, teacher_dim = activation_caching(lm_name=teacher.model_name,
                                               lm_hook_points=teacher.hook_list,
                                               dataset_name=dataset_name,
                                               dataset_subset=dataset_subset,
                                               split=split,
                                               seq_len=seq_len,
                                               batch_size=cache_batch_size,
                                               cache_dir=cache_dir,
                                               token_rename=token_rename,
                                               shuffle=shuffle,
                                               lm_dir=teacher_dir,
                                               total_tokens=total_tokens,
                                               force_cache_all=force_cache_all,
                                               cache_outputs=False,
                                               num_workers=num_workers,
                                               device=device)
    
    student_cache_dir, student_dim = activation_caching(lm_name=student.model_name,
                                               lm_hook_points=student.hook_list,
                                               dataset_name=dataset_name,
                                               dataset_subset=dataset_subset,
                                               split=split,
                                               seq_len=seq_len,
                                               batch_size=cache_batch_size,
                                               cache_dir=cache_dir,
                                               token_rename=token_rename,
                                               shuffle=shuffle,
                                               lm_dir=student_dir,
                                               total_tokens=total_tokens,
                                               force_cache_all=force_cache_all,
                                               cache_outputs=False,
                                               num_workers=num_workers,
                                               device=device)

    # create projector
    projector = Projector.create(
        teacher_cache_dir=teacher_cache_dir,
        student_cache_dir=student_cache_dir,
        teacher_dim=teacher_dim,
        student_dim=student_dim,
        alignment_beta=alignment_beta,
        max_epochs=num_epochs,
        weight_decay_choices=weight_decay_choices,
        max_grad_norm_choices=max_grad_norm_choices,
        batch_size_choices=batch_size_choices,
        n_trials=n_trials,
        save_dir=f"{save_dir}/projector_{alignment_beta}",
        seed=seed,
        num_gpus_per_trial=1.0,
        num_cpus_per_trial=8,
    )
    return projector
