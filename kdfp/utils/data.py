import os
import torch
from torch.utils.data import Dataset, ConcatDataset
from datasets import load_dataset, load_dataset_builder
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
            streaming=True,
        )

    else:
        dataset = load_dataset(
            dataset_name,
            split=split,
            trust_remote_code=True,
            streaming=True,
        )
    
    if n_samples:
        dataset = dataset.take(n_samples)

    preprocess_fn = partial(_preprocess,
                            seq_len=seq_len,
                            tokenizer=tokenizer,
                            token_rename=token_rename)
    dataset = dataset.map(
        preprocess_fn,
        batched=True,
    )

    if token_rename:
        dataset.set_format(type="torch", columns=["tokens"])
    return dataset


def _format_arc(examples, split: str):
    """
    Formats ARC examples for either training or NLL-based evaluation.

    Args:
        examples: A batch of examples from the ARC dataset.
        split (str): The dataset split, e.g., 'train' or 'validation'.
    """
    # Get the number of examples in the batch
    num_examples = len(examples['id'])
    
    if split == 'train':
        # For TRAINING: Create a single, correct question-answer pair.
        texts = []
        for i in range(num_examples):
            question = examples['question'][i]
            choices = examples['choices'][i]['text']
            choice_labels = examples['choices'][i]['label'] # e.g., ['A', 'B', 'C', 'D']
            answer_key = examples['answerKey'][i] # e.g., 'C'

            # Find the 0-indexed integer for the correct answer
            correct_idx = choice_labels.index(answer_key)
            correct_answer_text = choices[correct_idx]
            
            # Format as a complete and correct QA pair
            text = f"Question: {question}\nAnswer: {correct_answer_text}"
            texts.append(text)
            
        return texts, None # No separate labels needed for training

    else:
        # For VALIDATION: Create all possible QA pairs for each example
        # and return them in a flattened list for efficient batch tokenization.
        all_choice_texts = []
        labels = []
        for i in range(num_examples):
            question = examples['question'][i]
            choices = examples['choices'][i]['text']
            choice_labels = examples['choices'][i]['label']
            answer_key = examples['answerKey'][i]

            # Find the 0-indexed label
            label_idx = choice_labels.index(answer_key)
            labels.append(label_idx)

            # Create a formatted string for each of the choices
            for choice in choices:
                text = f"Question: {question}\nAnswer: {choice}"
                all_choice_texts.append(text)
            
        return all_choice_texts, labels

def _format_hellaswag(examples, split: str):
    """
    Prepares texts and labels for HellaSwag.

    For split='train': Returns a list of single correct sentences.
    For split='validation'/'test': Returns all 4 possible sentences for each example,
                                   flattened into a single list, plus the labels.
    """
    # Get the number of examples in the batch
    num_examples = len(examples['ctx'])
    
    if split == 'train':
        texts = []
        for i in range(num_examples):
            context = examples['ctx'][i]
            correct_ending = examples['endings'][i][int(examples['label'][i])]
            texts.append(context + " " + correct_ending)
        # No labels needed for training
        return texts, None
    else:
        # For validation, we create all 4 options for each example
        all_choice_texts = []
        labels = []
        for i in range(num_examples):
            context = examples['ctx'][i]
            for choice in examples['endings'][i]:
                all_choice_texts.append(context + " " + choice)
            labels.append(int(examples['label'][i]))
        return all_choice_texts, labels

def _format_piqa(examples, split: str):
    """
    Formats PIQA examples for either training or NLL-based evaluation.

    Args:
        examples: A batch of examples from the PIQA dataset.
        split (str): The dataset split, e.g., 'train' or 'validation'.
    """
    # Get the number of examples in the batch
    num_examples = len(examples['goal'])
    
    if split == 'train':
        # For TRAINING: Create a single, correct goal-solution pair.
        texts = []
        for i in range(num_examples):
            goal = examples['goal'][i]
            sol1 = examples['sol1'][i]
            sol2 = examples['sol2'][i]
            # The 'label' is already a 0-indexed integer (0 or 1)
            correct_label = examples['label'][i]
            correct_solution = sol1 if correct_label == 0 else sol2
            
            # Format as a complete and correct goal-solution pair
            text = f"Goal: {goal}\nSolution: {correct_solution}"
            texts.append(text)
            
        return texts, None # No separate labels needed for training

    else:
        # For VALIDATION: Create both possible goal-solution pairs for each example
        # and return them in a flattened list for efficient batch tokenization.
        all_choice_texts = []
        labels = []
        for i in range(num_examples):
            goal = examples['goal'][i]
            sol1 = examples['sol1'][i]
            sol2 = examples['sol2'][i]
            label_idx = examples['label'][i]
            
            labels.append(label_idx)

            # Create a formatted string for both possible solutions
            text1 = f"Goal: {goal}\nSolution: {sol1}"
            text2 = f"Goal: {goal}\nSolution: {sol2}"
            all_choice_texts.extend([text1, text2])
            
        return all_choice_texts, labels

def _format_boolq(examples, split: str):
    """
    Formats BoolQ examples for either training or NLL-based evaluation.

    Args:
        examples: A batch of examples from the BoolQ dataset.
        split (str): The dataset split, e.g., 'train' or 'validation'.
    """
    # Get the number of examples in the batch
    num_examples = len(examples['question'])
    
    # Define the fixed choices and their mapping from the boolean label
    # This ensures consistency: False -> 0 -> "no", True -> 1 -> "yes"
    choices = ["no", "yes"]

    if split == 'train':
        # For TRAINING: Create a single, correct question-answer pair.
        texts = []
        for i in range(num_examples):
            context = examples['passage'][i]
            question = examples['question'][i]
            # Determine the correct answer string based on the boolean label
            correct_answer_text = "yes" if examples['label'][i] else "no"
            
            # Format as a complete and correct QA pair
            text = f"Passage: {context}\nQuestion: {question}?\nAnswer: {correct_answer_text}"
            texts.append(text)
            
        return texts, None # No separate labels needed for training

    else:
        # For VALIDATION: Create both possible QA pairs for each example
        # and return them in a flattened list for efficient batch tokenization.
        all_choice_texts = []
        labels = []
        for i in range(num_examples):
            context = examples['passage'][i]
            question = examples['question'][i]
            
            # Determine the 0-indexed label (0 for "no", 1 for "yes")
            label_idx = 1 if examples['label'][i] else 0
            labels.append(label_idx)

            # Create a formatted string for both of the fixed choices ("no", "yes")
            for choice in choices:
                text = f"Passage: {context}\nQuestion: {question}?\nAnswer: {choice}"
                all_choice_texts.append(text)
            
        return all_choice_texts, labels

def _format_logiqa(examples, split: str):
    """
    Formats LogiQA examples for either training or NLL-based evaluation.

    Args:
        examples: A batch of examples from the LogiQA dataset.
        split (str): The dataset split, e.g., 'train' or 'validation'.
    """
    # Get the number of examples in the batch
    num_examples = len(examples['id'])
    
    if split == 'train':
        # For TRAINING: Create a single, correct question-answer pair.
        texts = []
        for i in range(num_examples):
            context = examples['text'][i]
            question = examples['question'][i]
            choices = examples["options"][i]
            # The 'answer' is already a 0-indexed integer
            correct_answer_idx = examples['answer'][i]
            correct_answer_text = choices[correct_answer_idx]
            
            # Format as a complete and correct QA pair
            text = f"Passage: {context}\nQuestion: {question}\nAnswer: {correct_answer_text}"
            texts.append(text)
            
        return texts, None # No separate labels needed for training

    else:
        # For VALIDATION: Create four possible QA pairs for each example
        # and return them in a flattened list for efficient batch tokenization.
        all_choice_texts = []
        labels = []
        for i in range(num_examples):
            context = examples['text'][i]
            question = examples['question'][i]
            choices = examples["options"][i]
            label_idx = examples['answer'][i]
            
            labels.append(label_idx)

            # Create a formatted string for each of the four choices
            for choice in choices:
                text = f"Passage: {context}\nQuestion: {question}\nAnswer: {choice}"
                all_choice_texts.append(text)
            
        return all_choice_texts, labels

def _format_sciq(examples, split: str):
    """
    Formats SciQ examples for either training or NLL-based evaluation.

    Args:
        examples: A batch of examples from the SciQ dataset.
        split (str): The dataset split, e.g., 'train' or 'validation'.
    """
    # Get the number of examples in the batch
    num_examples = len(examples['question'])
    
    if split == 'train':
        # For TRAINING: Create a single, correct question-answer pair.
        texts = []
        for i in range(num_examples):
            context = examples['support'][i]
            question = examples['question'][i]
            correct_answer = examples['correct_answer'][i]
            
            # Format as a complete and correct QA pair
            text = f"Context: {context}\nQuestion: {question}\nAnswer: {correct_answer}"
            texts.append(text)
            
        return texts, None # No separate labels needed for training

    else:
        # For VALIDATION: Create four possible QA pairs for each example
        # and return them in a flattened list for efficient batch tokenization.
        all_choice_texts = []
        labels = []
        for i in range(num_examples):
            context = examples['support'][i]
            question = examples['question'][i]
            correct_answer = examples['correct_answer'][i]
            
            # Assemble all four choices
            choices = [
                correct_answer, 
                examples["distractor1"][i], 
                examples["distractor2"][i], 
                examples["distractor3"][i]
            ]
            
            # Shuffle the choices to avoid positional bias
            random.shuffle(choices)
            
            # Find the new 0-indexed label of the correct answer
            label_idx = choices.index(correct_answer)
            labels.append(label_idx)

            # Create a formatted string for each of the four shuffled choices
            for choice in choices:
                text = f"Context: {context}\nQuestion: {question}\nAnswer: {choice}"
                all_choice_texts.append(text)
            
        return all_choice_texts, labels
    
def _format_winogrande(examples, split: str):
    """
    Formats Winogrande examples for either training or NLL-based evaluation.

    Args:
        examples: A batch of examples from the Winogrande dataset.
        split (str): The dataset split, e.g., 'train' or 'validation'.
    """
    # Get the number of examples in the batch
    num_examples = len(examples['sentence'])
    
    if split == 'train':
        # For TRAINING: Create a single, correctly filled-in sentence for each example.
        texts = []
        for i in range(num_examples):
            sentence = examples['sentence'][i]
            option1 = examples['option1'][i]
            option2 = examples['option2'][i]
            # The 'answer' in the dataset is 1-based, so we determine the correct text
            correct_option = option1 if examples['answer'][i] == '1' else option2
            
            # Replace the pronoun placeholder '_' with the correct option
            filled_sentence = sentence.replace('_', correct_option, 1)
            texts.append(filled_sentence)
            
        return texts, None # No separate labels needed for training

    else:
        # For VALIDATION: Create all possible sentences (2 for each example)
        # and return them in a flattened list for efficient batch tokenization.
        all_choice_texts = []
        labels = []
        for i in range(num_examples):
            sentence = examples['sentence'][i]
            option1 = examples['option1'][i]
            option2 = examples['option2'][i]
            
            # Create both possible sentences
            text_option1 = sentence.replace('_', option1, 1)
            text_option2 = sentence.replace('_', option2, 1)
            all_choice_texts.extend([text_option1, text_option2])
            
            # The 'answer' is 1 or 2; convert it to a 0-indexed label
            label_idx = int(examples['answer'][i]) - 1
            labels.append(label_idx)
            
        return all_choice_texts, labels

def _preprocess_task_ft(examples,
                        split: str,
                        task: TaskSpec,
                        seq_len: int,
                        tokenizer,):
    if "hellaswag" in task.name:
        texts, labels = _format_hellaswag(examples, split=split)
    elif "winogrande" in task.name:
        texts, labels = _format_winogrande(examples, split=split)
    elif "sciq" in task.name:
        texts, labels = _format_sciq(examples, split=split)
    elif "logiqa" in task.name:
        texts, labels = _format_logiqa(examples, split=split)
    elif "boolq" in task.name:
        texts, labels = _format_boolq(examples, split=split)
    elif "piqa" in task.name:
        texts, labels = _format_piqa(examples, split=split)
    elif "arc" in task.name:
        texts, labels = _format_arc(examples, split=split)
    else:
        raise ValueError(f"Task {task.name} is not currently supported.")
    

    # Tokenize all text inputs at once
    tokenized = tokenizer(
        texts,
        max_length=seq_len,
        padding='max_length',
        truncation=True,
        return_tensors="pt"
    )

    # if split == 'train':
    #     # For training, labels are the input_ids shifted
    #     tokenized['labels'] = tokenized['input_ids'].clone()
    #     return tokenized
    # else:
    if split != 'train':
        # For validation, reshape the tokenized outputs
        num_examples = len(labels)
        num_choices = 4  if task.task_type == 'multiclass' else 2

        # Reshape from [batch_size * 4, seq_len] to [batch_size, 4, seq_len]
        for key in tokenized:
            tokenized[key] = tokenized[key].view(num_examples, num_choices, -1)
        
        # Add the labels to the dictionary
        tokenized['label'] = torch.tensor(labels, dtype=torch.long)
    return tokenized


def load_task_data(
    task: "TaskSpec",
    tokenizer,
    seq_len: int,
    split: str,
    n_samples: Optional[int] = None,
) -> Dataset:
    """
    Loads and preprocesses a dataset for a given task.

    If task.test_available is False, it reserves 10% of the 'train' split
    for validation/testing purposes. `actual_split` is the split used for
    loading the data from HuggingFace, and `split` is used to decide
    how to preprocess and format the data.
    """
    actual_split = split

    # If a dedicated test set is not provided by the dataset, create a
    # holdout set from the training data.
    if not task.test_available:
        if split == "train":
            # When training, use the first 90% of the original training data.
            actual_split = f"train"
        elif split == "validation":
            # When 'test' or 'validation' is requested, use the last 10% of
            # the training data as a substitute validation set.
            actual_split = f"train"
        elif split == "test":
            actual_split = "validation"

    if task.subset:
        dataset = load_dataset(
            task.dataset_name,
            task.subset,
            split=actual_split,
            trust_remote_code=True,
            streaming=True,
        )
    else:
        dataset = load_dataset(
            task.dataset_name,
            split=actual_split,
            trust_remote_code=True,
            streaming=True,
        )
    if not task.test_available:
        if split == "train":
            dataset = dataset.take(task.ninety_percent)
        elif split == "validation":
            dataset = dataset.skip(task.ninety_percent)

    if n_samples:
        dataset = dataset.take(n_samples)

    preprocess_fn = partial(
        _preprocess_task_ft,
        split=split,  # the originally requested split name
        task=task,
        seq_len=seq_len,
        tokenizer=tokenizer,
    )

    torch.cuda.empty_cache()

    dataset = dataset.map(
        preprocess_fn,
        batched=True,
        remove_columns=dataset.column_names,
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
            # if not matches:
            #     warnings.warn(f"No hook points found for hook {hook_name}", RuntimeWarning)
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
    
