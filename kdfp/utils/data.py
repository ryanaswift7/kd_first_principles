import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from functools import partial
import pandas as pd
from tqdm.auto import tqdm
from typing import Optional, List
from pydantic import BaseModel, field_validator, model_validator


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
              token_rename: bool = False) -> Dataset:
    if subset:
        dataset = load_dataset(
            dataset_name,
            subset,
            split=split,
            trust_remote_code=True,
        )
    else:
        dataset = load_dataset(
            dataset_name,
            split=split,
            trust_remote_code=True,
        )

    # Use `partial` to pass `args` to `_preprocess_and_rename`
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
    resid: Optional[CachedActivationsDataset] = None
    resid_attn: Optional[CachedActivationsDataset] = None
    embed_attn: Optional[CachedActivationsDataset] = None
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
        # Validate we have exactly 3 hook points in the expected order
        if len(self.hook_list) not in [1, 3]:
            raise ValueError(
                "hook_list must contain either exactly 3 hook points in this order: resid, resid_attn, embed_attn"
                + "OR it must be a list with only the resid hook point (i.e. list of length 1 -> for SAE caching)"
            )
        
        try:
            self.resid = CachedActivationsDataset(
                cache_dir=os.path.join(self.cache_dir, self.hook_list[0]),
                batch_mapping_location=self.cache_dir
            )
            if len(self.hook_list) == 3:
                self.output_logits = CachedActivationsDataset(
                    cache_dir=os.path.join(self.cache_dir, "output_logits"),
                    batch_mapping_location=self.cache_dir
                )
                self.resid_attn = CachedActivationsDataset(
                    cache_dir=os.path.join(self.cache_dir, self.hook_list[1]),
                    batch_mapping_location=self.cache_dir
                )
                self.embed_attn = CachedActivationsDataset(
                    cache_dir=os.path.join(self.cache_dir, self.hook_list[2]),
                    batch_mapping_location=self.cache_dir
                )
        except Exception as e:
            raise ValueError(f"Failed to initialize datasets: {str(e)}")
        
        return self
    