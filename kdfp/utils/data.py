import os
import torch
from torch.utils.data import Dataset, ConcatDataset
from datasets import load_dataset
from functools import partial
import pandas as pd
from tqdm.auto import tqdm
from typing import Optional, List
from pydantic import BaseModel, field_validator, model_validator
from kdfp.schemas.tasks import TaskSpec
import random
import json
from pathlib import Path
import warnings
from filelock import FileLock

def _preprocess(examples, seq_len, tokenizer, token_rename: bool):
    # Preprocess the examples
    tokenized_examples = tokenizer(
        examples['text'], 
        return_tensors="pt", 
        truncation=True, 
        padding='max_length', 
        max_length=seq_len
    )
    
    # Rename 'input_ids' to 'tokens'
    if token_rename:
        tokenized_examples["tokens"] = tokenized_examples.pop("input_ids")
    
    return tokenized_examples

def load_data(tokenizer,
              seq_len: int,
              dataset_name: str,
              subset: Optional[str] = None,
              split: str = "train",
              token_rename: bool = False,
              n_samples: Optional[int] = None,
              streaming: bool = True) -> Dataset:
    if subset:
        dataset = load_dataset(
            dataset_name,
            subset,
            split=split,
            trust_remote_code=True,
            streaming=streaming,
        )
    else:
        dataset = load_dataset(
            dataset_name,
            split=split,
            trust_remote_code=True,
            streaming=streaming,
        )

    # Use `partial` to pass `args` to `_preprocess_and_rename`
    preprocess_fn = partial(_preprocess,
                            seq_len=seq_len,
                            tokenizer=tokenizer,
                            token_rename=token_rename)

    if n_samples:
        dataset = dataset.take(n_samples)

    dataset = dataset.map(
        preprocess_fn,
        batched=True,
    )

    if token_rename:
        dataset.set_format(type="torch", columns=["tokens"])
    return dataset


def _format_arc(examples):
    texts = []
    labels = []
    
    for i in range(len(examples["id"])):
        context = examples['question'][i]
        choices = examples['choices']['text'][i]
        choice_labels = examples['choices']['label'][i]
        ans = examples['answerKey'][i]
        if ans in ['A', 'B', 'C', 'D', 'E']:
            ans = ord(ans) - ord('A')
        elif ans in ['1', '2', '3', '4', '5']:
            ans = int(ans) - 1

        choice_list = [f"{choice_label}) {choice}" for choice_label, choice in zip(choice_labels, choices)]
        choice_text = "\n".join(choice_list)
        text = f"Question: {context}\nChoices: {choice_text}\nAnswer: "
        
        texts.append(text)
        labels.append(ans)
    return texts, labels

def _format_hellaswag(examples):
    texts = []
    labels = []
    
    for i in range(len(examples["ind"])):
        # HellaSwag format
        title = examples["activity_label"]
        context = examples['ctx'][i]
        choices = [examples['endings'][i][j] for j in range(4)]
        label = examples['label'][i]
        choice_list = [f"{idx}) {choice}" for idx, choice in enumerate(choices)]
        choice_text = "\n".join(choice_list)
        text = f"Title: {title}\nContext: {context}\nChoices: {choice_text}\nAnswer: "

        texts.append(text)
        labels.append(label)
    return texts, labels

def _format_piqa(examples):
    texts = []
    labels = []
    
    for i in range(len(examples["goal"])):
        context = examples['goal'][i]
        choices = [examples['sol1'][i], examples['sol2'][i]]
        label = examples['label'][i]
        choice_list = [f"{idx}) {choice}" for idx, choice in enumerate(choices)]
        choice_text = "\n".join(choice_list)
        text = f"Question: {context}\nChoices: {choice_text}\nAnswer: "
        
        texts.append(text)
        labels.append(label)
    return texts, labels

def _format_boolq(examples):
    texts = []
    labels = []
    
    for i in range(len(examples["idx"])):
        context = examples['passage'][i]
        question = examples['question'][i]
        label = examples['label'][i]
        text = f"Passage: {context}\nQuestion: {question}\nChoices: yes or no\nAnswer: "
        
        texts.append(text)
        labels.append(label)
    return texts, labels

def _format_logiqa(examples):
    texts = []
    labels = []
    
    for i in range(len(examples["id"])):
        context = examples['text'][i]
        question = examples['question'][i]
        raw_choices = examples["options"][i]
        choice_list = [f"A) {raw_choices[0]}", f"B) {raw_choices[1]}", f"C) {raw_choices[2]}", f"D) {raw_choices[3]}"]
        label = examples['answer'][i]
        choice_text = "\n".join(choice_list)
        text = f"Passage: {context}\nQuestion: {question}\nChoices: {choice_text}\nAnswer: "
        
        texts.append(text)
        labels.append(label)
    return texts, labels

def _format_sciq(examples):
    texts = []
    labels = []
    
    for i in range(len(examples["question"])):
        context = examples['support'][i]
        question = examples['question'][i]
        choices = [examples['correct_answer'][i], examples["distractor1"][i], examples["distractor2"][i], examples["distractor3"][i]]
        random.shuffle(choices)
        label = choices.index(examples['correct_answer'][i])
        choice_list = [f"{idx}) {choice}" for idx, choice in enumerate(choices)]
        choice_text = "\n".join(choice_list)
        text = f"Passage: {context}\nQuestion: {question}\nChoices: {choice_text}\nAnswer: "
        
        texts.append(text)
        labels.append(label)
    return texts, labels

def _format_winogrande(examples):
    texts = []
    labels = []
    
    for i in range(len(examples["sentence"])):
        context = examples['sentence'][i]
        idx = examples['sentence'][i].index('_') +1
        choices = [f"1) {examples['option1'][i] + context[idx:]}", f"2) {examples['option2'][i] + context[idx:]}"]
        label = examples['answer'][i]
        choice_text = "\n".join(choices)
        text = f"Passage: {context[:idx]}\nChoices: {choice_text}\nAnswer: "
        
        texts.append(text)
        labels.append(label)
    return texts, labels

def _preprocess_task_ft(examples,
                        task: TaskSpec,
                        seq_len: int,
                        tokenizer,):
    
    processed_examples = {}
    
    if 'arc' in task.name:
        texts, labels = _format_arc(examples)
    elif 'boolq' in task.name:
        texts, labels = _format_boolq(examples)
    elif 'hellaswag' in task.name:
        texts, labels = _format_hellaswag(examples)
    elif 'logiqa' in task.name:
        texts, labels = _format_logiqa(examples)
    elif 'piqa' in task.name:
        texts, labels = _format_piqa(examples)
    elif 'sciq'in task.name:
        texts, labels = _format_sciq(examples)
    elif 'winogrande' in task.name:
        texts, labels = _format_winogrande(examples)

    tokenized = tokenizer(
        texts,
        max_length=seq_len,
        padding='max_length',
        truncation=True,
        return_tensors="pt"
    )
    
    processed_examples.update(tokenized)
    processed_examples['labels'] = labels
    
    
    return processed_examples


def load_task_data(task: TaskSpec,
                   tokenizer,
                   seq_len: int,
                   split: str = 'train',
                #    token_rename: bool = False,
                   n_samples: Optional[int] = None,) -> Dataset:
    # task_types_map = {
    #     'hellaswag': 'multiple_choice',
    #     # 'trivia_qa': 'qa',
    #     'boolq': 'classification',
    #     'piqa': 'multiple_choice',
    #     'arc_easy': 'multiple_choice',
    #     'arc_challenge': 'multiple_choice',
    #     'winogrande': 'multiple_choice',
    #     'sciq': 'classification',
    #     'logiqa': 'multiple_choice',
    # }

    if task.subset:
        dataset = load_dataset(
            task.dataset,
            task.subset,
            split=split,
            trust_remote_code=True,
        )
    else:
        dataset = load_dataset(
            task.dataset,
            split=split,
            trust_remote_code=True,
        )
    
    # Limit samples if requested
    if n_samples and n_samples < len(dataset):
        dataset = dataset.select(range(n_samples))

    preprocess_fn = partial(
        _preprocess_task_ft,
        task=task,
        seq_len=seq_len,
        tokenizer=tokenizer,
        # task_type=task_types_map[task.name],
    )

    dataset = dataset.map(
        preprocess_fn,
        batched=True,
        remove_columns=dataset.column_names  # Remove original columns to save memory
    )
    return dataset
    


class CachedActivationsDataset(Dataset):
    """Dataset for working with cached activations from TransformerLens."""
    
    def __init__(self, cache_dir, batch_mapping_location: Optional[str] = None, target_data_path=None, preload=False, transform=None):
        """
        Args:
            cache_dir: Directory containing the cached activation files
            target_data_path: Path to targets/labels (CSV or pickle format)
            preload: Whether to preload all activations into memory (faster but memory-intensive)
            transform: Optional transform to apply to the activations
        """
        self.cache_dir = cache_dir
        self.transform = transform
        
        # Load batch mapping information
        if not batch_mapping_location:
            batch_mapping_location = cache_dir
        mapping_path = os.path.join(batch_mapping_location, "batch_mapping.txt")
        self.batch_mapping = pd.read_csv(mapping_path)
        
        # Create a lookup from sample index to batch file
        self.sample_to_batch = {}
        self.sample_to_position = {}
        for _, row in self.batch_mapping.iterrows():
            batch_idx = row['batch_idx']
            start_idx = row['start_sample_idx']
            end_idx = row['end_sample_idx']
            
            for sample_idx in range(start_idx, end_idx):
                self.sample_to_batch[sample_idx] = batch_idx
                # Position within the batch
                self.sample_to_position[sample_idx] = sample_idx - start_idx
        
        # Get total number of samples
        self.num_samples = max(self.sample_to_batch.keys()) + 1
        
        # Load targets/labels if provided
        self.targets = None
        if target_data_path:
            if target_data_path.endswith('.csv'):
                self.targets = pd.read_csv(target_data_path)
            elif target_data_path.endswith('.pkl'):
                self.targets = pd.read_pickle(target_data_path)
            else:
                raise ValueError("Target data must be CSV or pickle format")
        
        # Check the format of the activation tensors by loading the first batch
        first_batch_path = os.path.join(cache_dir, f"batch_{0:05d}.pt")
        first_batch = torch.load(first_batch_path, weights_only=True)
        self.activation_shape = first_batch.shape[1:]  # Exclude batch dimension
        
        # Preload all activations if specified
        self.preloaded_activations = {}
        if preload:
            print("Preloading activations into memory...")
            unique_batches = set(self.sample_to_batch.values())
            for batch_idx in tqdm(unique_batches):
                batch_path = os.path.join(cache_dir, f"batch_{batch_idx:05d}.pt")
                self.preloaded_activations[batch_idx] = torch.load(batch_path, weights_only=True)
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # Get the batch this sample belongs to
        batch_idx = self.sample_to_batch[idx]
        position = self.sample_to_position[idx]
        
        # Get activation - either from preloaded memory or disk
        if batch_idx in self.preloaded_activations:
            batch_activations = self.preloaded_activations[batch_idx]
        else:
            batch_path = os.path.join(self.cache_dir, f"batch_{batch_idx:05d}.pt")
            batch_activations = torch.load(batch_path, weights_only=True)
        
        # Extract the specific activation for this sample
        # Handling different possible formats from TransformerLens
        if position < batch_activations.shape[0]:
            activation = batch_activations[position]
        else:
            raise IndexError(f"Position {position} out of bounds for batch with shape {batch_activations.shape}")
        
        # Apply transform if specified
        if self.transform:
            activation = self.transform(activation)
        
        # Get target if available
        if self.targets is not None:
            target = torch.tensor(self.targets.iloc[idx]['target'])
            return activation, target
        else:
            return activation


class HookCache(BaseModel):
    cache_dir: str
    hook_list: List[str]
    embed: Optional[CachedActivationsDataset] = None
    attn_scores: Optional[List[CachedActivationsDataset]] = None
    attn_out: Optional[List[CachedActivationsDataset]] = None
    mlp_out: Optional[List[CachedActivationsDataset]] = None
    resid: Optional[List[CachedActivationsDataset]] = None
    output_logits: Optional[CachedActivationsDataset] = None

    model_config = {
        "extra": "forbid",  # Raises error on unrecognized fields
        "arbitrary_types_allowed": True,
    }

    @field_validator('cache_dir')
    def ensure_cache_exists(cls, v: str) -> str:
        """Validate that the cache directory exists."""
        if not os.path.isdir(v):
            raise ValueError(f"Directory {v} does not exist.")
        return v
    
    @field_validator('hook_list', mode='after')
    def ensure_hooks_in_cache(cls, v: List[str], values) -> List[str]:
        """Validate that all hook directories exist in the cache."""
        if 'cache_dir' not in values.data:
            return v  # cache_dir validation failed, skip this check
            
        cache_dir = values.data['cache_dir']
        for hp in v:
            if not os.path.isdir(os.path.join(cache_dir, hp)):
                raise ValueError(f"Cache for hook {hp} not found in {cache_dir}.")
        return v
    
    @model_validator(mode='after')
    def setup_datasets(self) -> 'HookCache':
        """Initialize the dataset objects after validation."""

        # collect and organize hooks
        hook_dict = {
            "embed": [],
            "attn_scores": [],
            "attn_out": [],
            "mlp_out": [],
            "resid": [],
            "output_logits": [],
        }

        for hook_name in hook_dict.keys():
            matches = [s for s in self.hook_list if hook_name in s]
            if not matches:
                warnings.warn(f"No hook points found for hook {hook_name}", RuntimeWarning)
            hook_dict[hook_name] = matches

        # run verifications
        all_matched_hooks = [hook for hook_list in hook_dict.values() for hook in hook_list]

        if len(all_matched_hooks) != len(self.hook_list):
            unrecognized_hooks = set(self.hook_list) - set(all_matched_hooks)
            raise ValueError(f"Some hooks were not recognized: {unrecognized_hooks}")
        
        def get_layer_from_hook_name(hook_name: str) -> int:
            # ex) blocks.0.hook_resid_post, or blocks.0.attn.hook_attn_scores
            return hook_name.split('.')[1]
        
        def form_cached_dataset_lists(hook_type: str) -> None:
            if hook_dict[hook_type]:
                ds_list = []
                for layer_hook_name in hook_dict[hook_type]:
                    ds_list.append(CachedActivationsDataset(
                        cache_dir=os.path.join(self.cache_dir, layer_hook_name),
                        batch_mapping_location=self.cache_dir))
            return ds_list
        
        try:
            layer_hook_type_list = ["attn_scores", "attn_out", "mlp_out", "resid"]
            for hook_type in layer_hook_type_list:
                ds_list = form_cached_dataset_lists(hook_type=hook_type)
                setattr(self, hook_type, ds_list)

            if hook_dict["embed"]:
                self.embed = CachedActivationsDataset(
                        cache_dir=os.path.join(self.cache_dir, "hook_embed"),
                        batch_mapping_location=self.cache_dir)
                
            if hook_dict["output_logits"]:
                self.output_logits = CachedActivationsDataset(
                        cache_dir=os.path.join(self.cache_dir, "output_logits"),
                        batch_mapping_location=self.cache_dir)
            n = len(self.attn_scores)
            for hook_type in layer_hook_type_list:
                v = getattr(self, hook_type, [])
                assert len(v) == n, f"Hook type {hook_type} contains {len(v)} layers/datasets, expected {n}"

        except Exception as e:
            raise ValueError(f"Failed to initialize datasets: {str(e)}")
        
        return self
    


class ProjectorDataset(Dataset):
    def __init__(self, teacher_cache_dir: str, student_cache_dir: str):

        teacher_cache_dir = os.path.abspath(teacher_cache_dir)
        student_cache_dir = os.path.abspath(student_cache_dir)

        # Use a single lock for both directories to prevent deadlocks
        # lock_path = f"/tmp/projector_dataset_{hash(teacher_cache_dir + student_cache_dir)}.lock"
        
        # with FileLock(lock_path):
        #     teacher_ds = self._get_projector_dataset(teacher_cache_dir)
        #     student_ds = self._get_projector_dataset(student_cache_dir)

        # teacher_len = len(teacher_ds)
        # student_len = len(student_ds)
        # num_rows = min(teacher_len, student_len)
        # indices = list(range(num_rows))

        # self.num_rows = num_rows
        # self.teacher_ds = teacher_ds
        # self.student_ds = student_ds
        self.teacher_cache_dir = teacher_cache_dir
        self.student_cache_dir = student_cache_dir

        # Just determine the length, don't load datasets yet
        self._teacher_ds = None
        self._student_ds = None
        self.num_rows = self._get_num_rows()

    def lock(self) -> str:
        # Use a single lock for both directories to prevent deadlocks
        lock_path = f"/tmp/projector_dataset_{hash(self.teacher_cache_dir + self.student_cache_dir)}.lock"
        
        # with FileLock(lock_path):
        return lock_path

    def _get_num_rows(self):
        # with FileLock(self.lock()):
        # Get length without fully loading datasets
        teacher_ds = self._get_projector_dataset(self.teacher_cache_dir)
        student_ds = self._get_projector_dataset(self.student_cache_dir)
        return min(len(teacher_ds), len(student_ds))

    @property
    def teacher_ds(self):
        if self._teacher_ds is None:
            self._teacher_ds = self._get_projector_dataset(self.teacher_cache_dir)
        return self._teacher_ds

    @property
    def student_ds(self):
        if self._student_ds is None:
            self._student_ds = self._get_projector_dataset(self.student_cache_dir)
        return self._student_ds

    def _get_projector_dataset(self, cache_dir: str):
        # with FileLock(self.lock()):
        # ignore attn_scores and output_logits bc they are not in the embedding space
        with open(Path(cache_dir)/"caching_summary.json", "r") as f:
            data = json.load(f)
        cached_hook_list = data["lm_hook_points"]
        hook_cache = HookCache(cache_dir=cache_dir, hook_list=cached_hook_list)
        dataset_list = []
        if hook_cache.embed:
            dataset_list.append(hook_cache.embed)
        for act_list in [hook_cache.attn_out, hook_cache.mlp_out, hook_cache.resid]:
            if act_list:
                for layer in act_list:
                    dataset_list.append(layer)
        return ConcatDataset(datasets=dataset_list)
    
    def __len__(self):
        return self.num_rows
    
    def __getitem__(self, index):
        X_teacher = self.teacher_ds[index]
        X_student = self.student_ds[index]
        return X_teacher, X_student
    
