import os
import torch
# from sae_lens import (
#     HookedSAETransformer,
#     SAE,
# )
from transformer_lens import HookedTransformer
from tqdm.auto import tqdm
import numpy as np
from torch.utils.data import DataLoader
from datasets import load_dataset, Dataset
from functools import partial
from pathlib import Path
import sys
from .general import set_vram_limit, set_all_seeds, cleanup_resources, print_args_table, set_device
from .data import CachedActivationsDataset, load_data, HookCache
import argparse
import sys
import math
import pandas as pd
from typing import List, Optional
import json


def parse_args():
    parser = argparse.ArgumentParser(
        description="Cache HookedTransformer activations for a dataset",
    )
    # parser.add_argument(
    #     "--cache-mode",
    #     type=str,
    #     choices=["lm", "sae", "both"],
    #     help="Which activations to cache: 'lm' or 'sae' or 'both'.",
    #     required=True,
    # )
    parser.add_argument(
        "--lm-name",
        type=str,
        help="Name of HookedTransformer model to use",
        required=True,
    )
    parser.add_argument(
        "--lm-dir",
        type=str,
        help="Directory containing LM's 'state_dict.pt'.",
        required=True,
    )
    parser.add_argument(
        "--lm-act-cache-dir",
        type=str,
        help="Directory where you would like to save the activations.",
        required=True,
    )
    # parser.add_argument(
    #     "--sae-dir",
    #     type=str,
    #     default=None,
    #     help="Directory where SAE is saved to.",
    # )
    # parser.add_argument(
    #     "--cache-subdir",
    #     type=str,
    #     help="Directory for the language model activation cache. For example, LM activations will be placed under model_dir/cache_subdir/lm_acts.",
    #     required=True,
    # )
    # parser.add_argument(
    #     "--unmodified",
    #     action="store_true",
    #     default=False,
    #     help="Flag to indicate using unmodified model (i.e. baseline/student or teacher).",
    # )
    parser.add_argument("--dataset-name", type=str, required=True, help="HuggingFace dataset to use")
    parser.add_argument("--dataset-subset", type=str, default=None, help="Subset of main dataset to use")
    parser.add_argument("--split", type=str, default='train', help="Dataset split to use (i.e. train, val, dev, test, etc.)")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for processing")
    parser.add_argument("--total-tokens", type=int, default=None, help="Total number of tokens to cache activations for.")
    parser.add_argument("--seq-len", type=int, default=512, help="Number of tokens to include in each sample sequence.")
    parser.add_argument(
        "--shuffle",
        action="store_true",
        default=False,
        help="Use this flag if you want the data to be shuffled.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=44,
        help="Set the random seed value.",
    )
    parser.add_argument(
        "--reproducible-backend",
        action="store_true",
        default=False,
        help="Use this flag if you need full reproducbility (i.e. in CUDA).",
    )
    parser.add_argument(
        "--vram-limit-gb",
        type=float,
        default=None,
        help="Set amount VRAM (in GB) to allow CUDA to use.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Set CUDA device to use.",
    )
    parser.add_argument(
        "--lm-hook-points",
        type=str,
        required=True,
        help="Comma separated string of hook points to cache in LM.",
    )
    # parser.add_argument(
    #     "--sae-hook-point",
    #     type=str,
    #     default="hook_sae_acts_post",
    #     help="Comma separated string of hook points to cache in SAE.",
    # )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Force re-caching even if existing cache appears complete",
    )
    parser.add_argument(
        "--cache-outputs",
        action="store_true",
        default=False,
        help="Cache LM outputs in addition to hidden activations. Doesn't apply to SAE caching",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=None,
        help="Set a maximum number of data samples to cache activations for.",
    )
    parser.add_argument(
        "--token-rename",
        action="store_true",
        default=False,
        help="Rename 'input_ids' to 'tokens' after tokenization. Needed (?) for training HookedTransformers",
    )
    parser.add_argument(
        "--dl-num-workers",
        type=int,
        default=None,
        help="Number of workers for dataloaders.",
    )


    return parser.parse_args()

def prepare_lm_and_dataloader(lm_name: str,
                              dataset_name: str,
                              dataset_subset: Optional[str],
                              split: str,
                              seq_len: int,
                              batch_size: int,
                              torch_dtype: torch.dtype,
                              token_rename: bool = False,
                              shuffle: bool = False,
                              lm_dir: Optional[str] = None,
                              total_tokens: Optional[int] = None,
                              num_workers: Optional[int] = None,):
    """Load model, sae, and dataset based on args."""
    if lm_dir:
        print(f"Loading model {lm_name} from local version in {lm_dir}")

    # N_DEVICES = 1

    # # may be the case that there is no sae yet (lm training)
    # sae = None
    # if args.cache_mode in ["sae", "both"]:
    #     sae = SAE.load_from_pretrained(args.sae_dir)
    #     lm = HookedSAETransformer.from_pretrained_no_processing(
    #                     args.model_name,
    #                     **sae.cfg.model_from_pretrained_kwargs,
    #                     n_devices=N_DEVICES,
    #                     )
    # else:
    lm = HookedTransformer.from_pretrained_no_processing(lm_name, dtype=torch_dtype)  #, n_devices=N_DEVICES)
    tokenizer = lm.tokenizer

    if lm_dir:
        lm.load_state_dict(torch.load(f"{lm_dir}/state_dict.pt", weights_only=True))

    # only load weights for modified models
    # if not args.unmodified:
    #     # not strict because projector parameters are saved here as well for online students
    #     lm_weights = torch.load(f"{args.lm_dir}/weights_final.pt", weights_only=True)
    #     lm.load_state_dict(lm_weights, strict=False)

    # load selected number of data samples
    n_samples = math.floor(total_tokens / seq_len) if total_tokens else None

    dataset = load_data(dataset_name=dataset_name,
                        subset=dataset_subset,
                        split=split,
                        tokenizer=tokenizer,
                        seq_len=seq_len,
                        token_rename=token_rename,
                        n_samples=n_samples,)
    dl = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )

    return lm, dl, n_samples

def create_caching_hook(batch_idx, save_dir):
    """Create a hook function that saves activations to disk."""
    def caching_hook(act, hook):
        # Create a filename based on batch index
        filename = os.path.join(save_dir, f"batch_{batch_idx:05d}.pt")
        
        # Move activation to CPU and detach from computation graph
        act_cpu = act.detach().cpu()
        
        # Save the activation tensor
        torch.save(act_cpu, filename)
        
        # Return the activation unchanged for the forward pass
        return act
    
    return caching_hook

def cache_activations(
        model,
        dataloader,
        cache_outputs: bool,
        hook_points: List[str],
        save_dir,
        device,
        n_samples: Optional[int],
        token_rename: bool = False,):
    """Process dataset and cache activations at specified hook point."""

    print("N SAMPLES: ", n_samples)
    print(f"\nCaching activations at:")
    for hp in hook_points:
        print(hp)    
    # Create a mapping file to track what's in each batch
    batch_map = []
    total_samples_cached = 0

    if cache_outputs:
        output_logits_dir = f"{save_dir}/output_logits"
        os.makedirs(output_logits_dir, exist_ok=True)

    
    model.to(device)
    try:
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader)):
                if token_rename:
                    input_ids = batch["tokens"].to(device)
                else:
                    input_ids = batch["input_ids"].to(device)

                # Check if we need to truncate this batch to meet n_samples limit
                current_batch_size = input_ids.shape[0]
                if n_samples is not None:
                    remaining_samples = n_samples - total_samples_cached
                    if remaining_samples <= 0:
                        break  # We've already cached enough samples
                    if current_batch_size > remaining_samples:
                        # Truncate the batch to only process the remaining samples needed
                        input_ids = input_ids[:remaining_samples]
                        current_batch_size = remaining_samples

                # create full list of hooks and set their cache dirs
                fwd_hooks = [(hp, create_caching_hook(batch_idx, f"{save_dir}/{hp}")) for hp in hook_points]
                
                # Register the hook for caching
                output_logits = model.run_with_hooks(
                    input_ids,
                    fwd_hooks=fwd_hooks,
                )

                if cache_outputs:
                    logits_filename = os.path.join(output_logits_dir, f"batch_{batch_idx:05d}.pt")
                    torch.save(output_logits.detach().cpu(), logits_filename)
                
                # Save mapping information (what samples are in this batch)
                start_idx = total_samples_cached
                end_idx = start_idx + current_batch_size
                batch_map.append((batch_idx, start_idx, end_idx))
                total_samples_cached += current_batch_size
                
                # Break if we've reached the desired number of samples
                if n_samples is not None and total_samples_cached >= n_samples:
                    break

        # Save the batch mapping for future reference
        with open(os.path.join(save_dir, "batch_mapping.txt"), "w") as f:
            f.write("batch_idx,start_sample_idx,end_sample_idx\n")
            for batch_idx, start_idx, end_idx in batch_map:
                f.write(f"{batch_idx},{start_idx},{end_idx}\n")
        
        print(f"Finished caching {len(batch_map)} batches with {total_samples_cached} total samples.")
        if n_samples is not None:
            print(f"Target samples: {n_samples}, Actual samples cached: {total_samples_cached}")

    except RuntimeError as e:
        if 'out of memory' in str(e).lower():
            print("Your process hit the memory limit - exiting gracefully")
            cleanup_resources()
            sys.exit(1)
        else:
            print(e)
            raise
    except Exception as e:
        print(f"Error in run_with_hooks: {e}")
        raise

def is_cache_complete(
        cache_dir: str,
        # expected_batches: int,
        n_samples: Optional[int] = None) -> bool:
    """Check if cache directory contains all expected batches and samples."""
    if not os.path.exists(cache_dir):
        print(f"Directory {cache_dir} not found.")
        return False
    
    # Check mapping file
    mapping_file = os.path.join(Path(cache_dir).parent, "batch_mapping.txt")
    if not os.path.exists(mapping_file):
        # print(f"Mapping file not found in {Path(cache_dir).parent}.")
        return False
    
    # If n_samples is specified, check if we have at least that many samples
    if n_samples is not None:
        try:
            with open(mapping_file, "r") as f:
                lines = f.readlines()[1:]  # Skip header
                total_samples = 0
                for line in lines:
                    parts = line.strip().split(',')
                    if len(parts) >= 3:
                        start_idx = int(parts[1])
                        end_idx = int(parts[2])
                        total_samples = max(total_samples, end_idx)
                
                if total_samples < n_samples:
                    # print(f"Error: Cache contains only {total_samples} samples, but {n_samples} samples are required.")
                    return False
                # else:
                #     print(f"Cache contains {total_samples} samples (required: {n_samples}).")
                    
        except Exception as e:
            print(f"Error reading mapping file: {e}")
            return False
    
    return True


def activation_caching(lm_name: str,
                       lm_hook_points: List[str],
                       dataset_name: str,
                       dataset_subset: Optional[str],
                       split: str,
                       seq_len: int,
                       batch_size: int,
                       cache_dir: str,
                       torch_dtype: torch.dtype,
                       token_rename: bool = False,
                       shuffle: bool = False,
                       lm_dir: Optional[str] = None,
                       total_tokens: Optional[int] = None,
                       num_workers: Optional[int] = None,
                       force_cache_all: bool = False,
                       cache_outputs: bool = False,
                       device: str = "cuda" if torch.cuda.is_available() else 'cpu') -> str:
    
    # resid_hooks = list(filter(lambda s: "hook_resid_post" in s , lm_hook_points))
    # sae_resid_hook_item = resid_hooks[round(len(resid_hooks) / 2)]

    # if args.unmodified:
    #     args.cache_dir = f"{args.model_dir}/{args.cache_subdir}"
    # else:
    #     mdir = Path(args.model_dir)
    #     args.cache_dir = str(mdir.parent / "cache")
    lm_act_cache_dir = f"{cache_dir}/{lm_name.split('/')[1]}/{dataset_name.split('/')[1]}"

    lm, token_dl, n_samples = prepare_lm_and_dataloader(
        lm_name=lm_name,
        dataset_name=dataset_name,
        dataset_subset=dataset_subset,
        split=split,
        seq_len=seq_len,
        batch_size=batch_size,
        torch_dtype=torch_dtype,
        token_rename=token_rename,
        shuffle=shuffle,
        lm_dir=lm_dir,
        total_tokens=total_tokens,
        num_workers=num_workers,
    )
    # if sae:
    #     assert resid_hook_item == sae.cfg.hook_name, f"resid hook: {resid_hook_item}\nsae cfg hook: {sae.cfg.hook_name}"

    # if args.unmodified:
    #     args.lm_cache_dir = f"{args.lm_dir}/{args.cache_subdir}"
    # else:
    #     lmdir = Path(args.lm_dir)
    #     args.lm_cache_dir = str(lmdir.parent / "cache")

    # lm_act_cache_dir = f"{args.lm_cache_dir}/lm_acts"
    # sae_act_cache_dir = f"{args.sae_dir}/sae_acts"

    # Model activations caching
# if args.cache_mode in ["lm", "both"]:
    hook_caches = [f"{lm_act_cache_dir}/{hp}" for hp in lm_hook_points]

    if force_cache_all:
        hooks_needed = lm_hook_points
    else:
        hooks_needed = []
        for idx, cache in enumerate(hook_caches):
            os.makedirs(cache, exist_ok=True)
        
            if not is_cache_complete(cache, n_samples=n_samples):
                hooks_needed.append(lm_hook_points[idx])

    if hooks_needed:
        print("\nStarting model activation caching...")
        print(f"Requested hook points:")
        for hp in lm_hook_points:
            print(hp)
        if hooks_needed != lm_hook_points:
            print("\nSome hooks have already been cached!")
        else:
            print("\nNo existing cache found. Caching all requested hook points.")
        # print("Caching activations for:")
        # print(f"{hp}\n" for hp in hooks_needed)
        cache_activations(
            model=lm,
            dataloader=token_dl,
            cache_outputs=cache_outputs,
            hook_points=hooks_needed,  # args.hook_point,
            save_dir=lm_act_cache_dir,
            device=device,
            n_samples=n_samples,
            token_rename=token_rename,
        )  
        print(f"\nModel activation caching complete. Batch file for all hook points saved at {lm_act_cache_dir}/batch_mapping.txt")
    else:
        print(f"\nUsing existing lm activations cache at {lm_act_cache_dir}.")
    
    # args.lm_act_cache_dir = lm_act_cache_dir
    # else:
    #     print("Skipping model activation caching as per --cache_mode")
        # args.lm_act_cache_dir = None

    # # SAE activations caching
    # if args.cache_mode in ["sae", "both"]:
    #     # if args.lm_act_cache_dir is None:
    #     #     lm_act_cache_dir = f"{args.cache_dir}/lm_acts"
    #     #     if not os.path.exists(lm_act_cache_dir):
    #     #         raise FileNotFoundError(
    #     #             "Cannot run SAE-only caching without existing model activations."
    #     #         )
    #     #     args.lm_act_cache_dir = lm_act_cache_dir

    #     # sae_act_cache_dir = f"{args.cache_dir}/sae_acts"
    #     sae_hook_point = "hook_sae_acts_post"
    #     sae_cache = f"{sae_act_cache_dir}/{sae_hook_point}"
    #     os.makedirs(sae_cache, exist_ok=True)
    #     cache_needed = args.force or not is_cache_complete(sae_cache, args.n_batches)
        
    #     if cache_needed:
    #         # lm_acts = CachedActivationsDataset(f"{args.lm_act_cache_dir}/{resid_hook_item}")
    #         lm_acts = HookCache(cache_dir=args.lm_act_cache_dir, hook_list=resid_hook)
    #         lm_act_dl = DataLoader(
    #             lm_acts.resid,
    #             batch_size=args.batch_size,
    #             shuffle=args.shuffle,
    #         )

    #         print("Starting SAE activation caching...")
    #         cache_activations(
    #             model=sae,
    #             dataloader=lm_act_dl,
    #             cache_outputs=False,
    #             hook_points=[sae_hook_point],
    #             save_dir=sae_act_cache_dir,
    #             device=args.device,
    #             sae=True,
    #             n_samples=args.n_samples,
    #         )
    #         print(f"SAE activation caching complete. Files saved to {sae_act_cache_dir}")
    #     else:
    #         print(f"Using existing SAE activations cache at {sae_act_cache_dir}")
        
    #     args.sae_act_cache_dir = sae_act_cache_dir
    # else:
    #     print("Skipping SAE activation caching as per --cache_mode")
    #     args.sae_act_cache_dir = None

    cached_tokens = n_samples * seq_len

    summary = {
        "lm_hook_points": lm_hook_points,
        "lm_name": lm_name,
        "dataset_name": dataset_name,
        "dataset_subset": dataset_subset,
        "split": split,
        "seq_len": seq_len,
        "batch_size": batch_size,
        "lm_act_cache_dir": lm_act_cache_dir,
        "token_rename": token_rename,
        "shuffle": shuffle,
        "lm_dir": lm_dir,
        "total_requested_tokens": total_tokens,
        "total_cached_tokens": cached_tokens,
        "num_workers": num_workers,
        "force_cache_all": force_cache_all,
        "cache_outputs": cache_outputs,
        "device": device,
    }


    # print("====== SUMMARY ======")
    # table = print_args_table(args)
    
    # Save summary to appropriate directory
    # output_dir = args.sae_act_cache_dir if args.sae_act_cache_dir else args.lm_act_cache_dir
    with open(f"{lm_act_cache_dir}/caching_summary.json", 'w') as f:
        json.dump(summary, f, indent=4)

    return lm_act_cache_dir, lm.cfg.d_model  #, args.sae_act_cache_dir

if __name__ == "__main__":
    args = parse_args()

    set_device(args.device)
    set_all_seeds(seed=args.seed, reproducible_backend=args.reproducible_backend)
    if args.vram_limit_gb:
        set_vram_limit(gb_limit=args.vram_limit_gb)
    
    all_lm_hook_points = args.lm_hook_points.split(',')

    force_cache_all = args.force
    num_workers = args.dl_num_workers


    activation_caching(lm_hook_points=all_lm_hook_points,
         lm_name=args.lm_name,
         dataset_name=args.dataset_name,
         dataset_subset=args.dataset_subset,
         split=args.split,
         seq_len=args.seq_len,
         batch_size=args.batch_size,
         lm_act_cache_dir=args.lm_act_cache_dir,
         token_rename=args.token_rename,
         shuffle=args.shuffle,
         lm_dir=args.lm_dir,
         total_tokens=args.total_tokens,
         num_workers=args.num_workers,
         force_cache_all=args.force_cache_all,
         cache_outputs=args.cache_outputs,
         device=args.device)