import os
import json
import torch
import gc
import lm_eval
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional
from transformers import AutoModelForCausalLM, AutoConfig


def _format_params(num_params: int) -> str:
    """Formats a number into a human-readable string (e.g., 1.2M, 2.5B)."""
    if num_params >= 1_000_000_000:
        return f"{num_params / 1_000_000_000:.2f}B"
    elif num_params >= 1_000_000:
        return f"{num_params / 1_000_000:.2f}M"
    elif num_params >= 1_000:
        return f"{num_params / 1_000:.2f}K"
    else:
        return f"{num_params}"

def extract_task_metrics(task_name: str, metrics: dict) -> dict:
    """Extracts and normalizes valid metrics for a given task."""
    if "bbh" in task_name and task_name != "bbh": return {}
    if "mmlu" in task_name and task_name != "mmlu": return {}

    acc = metrics.get("acc,none", metrics.get("acc"))
    norm = metrics.get("acc_norm,none", metrics.get("acc_norm"))
    em = metrics.get("exact_match,get-answer", metrics.get("exact_match,remove_whitespace", metrics.get("exact_match,none", metrics.get("exact_match"))))
    f1 = metrics.get("f1,none", metrics.get("f1"))
    inst = metrics.get("inst_level_strict_acc,none", metrics.get("inst_level_strict_acc"))

    task_scores = {}
    if acc is not None: task_scores["acc"] = acc
    if norm is not None: task_scores["acc_norm"] = norm
    if em is not None: task_scores["exact_match"] = em
    if f1 is not None: task_scores["f1"] = f1
    if inst is not None: task_scores["inst_level_strict_acc"] = inst

    return task_scores

def generate_result(
    name: str,
    total_params_str: str,
    trainable_params_str: str,
    task_name: str,
    metric_name: str,
    score: float,
    num_fewshot: Optional[int] = None,
) -> Dict:
    """Generates a flat dictionary row for the pandas DataFrame."""
    return {
        "EXPERIMENT": name,
        "TOTAL_PARAMS": total_params_str,
        "TRAINABLE_PARAMS": trainable_params_str,
        "TASK": task_name,
        "METRIC": metric_name,
        "NUM_FEWSHOT": num_fewshot,
        "SCORE": score,
    }

def evaluate_model(
    model_path: str, 
    tasks: list, 
    experiment_name: str,
    num_fewshot: int = None, 
    batch_size="auto", 
    limit: int = None, 
    device: str = None
) -> tuple[dict, pd.DataFrame, dict]:
    """
    Runs evaluation, extracts metrics, and returns formatting dicts and DataFrames.
    Loads model strictly onto the 'meta' device for instant, 0-memory parameter counting.
    """
    
    # 1. Fast Parameter Calculation (Meta Device)
    print(f"\nCalculating parameters for {experiment_name} via meta device...")
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    
    with torch.device("meta"):
        skeleton_model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
        
    total_params = sum(p.numel() for p in skeleton_model.parameters())
    trainable_params = sum(p.numel() for p in skeleton_model.parameters() if p.requires_grad)
    
    total_params_str = _format_params(total_params)
    trainable_params_str = _format_params(trainable_params)
    
    del skeleton_model
    gc.collect()

    # 2. Run lm_eval
    eval_args = {
        "model": "hf",  # "vllm",
        "model_args": f"pretrained={model_path},dtype=auto",  # ,gpu_memory_utilization=0.8",
        "tasks": tasks,
        "batch_size": batch_size,
    }
    if num_fewshot is not None: eval_args["num_fewshot"] = num_fewshot
    if limit is not None: eval_args["limit"] = limit
    if device is not None: eval_args["device"] = device

    print(f"\n--- Running Evaluation on {len(tasks)} tasks ---")
    raw_eval_output = lm_eval.simple_evaluate(**eval_args)
    raw_results_dict = raw_eval_output.get("results", {})

    extracted_scores_dict = {}
    results_list = []

    # 3. Process Results
    if raw_results_dict:
        print("\nEvaluation Results:")
        for task, raw_metrics in raw_results_dict.items():
            
            task_scores = extract_task_metrics(task, raw_metrics)
            if not task_scores:
                continue
            
            extracted_scores_dict[task] = task_scores

            # Terminal Print
            print_str = f"{task:<20} | "
            if "acc_norm" in task_scores: print_str += f"Norm: {task_scores['acc_norm']:.4f}  "
            if "acc" in task_scores: print_str += f"Acc: {task_scores['acc']:.4f}  "
            if "exact_match" in task_scores: print_str += f"EM: {task_scores['exact_match']:.4f}  "
            if "f1" in task_scores: print_str += f"F1: {task_scores['f1']:.4f}  "
            if "inst_level_strict_acc" in task_scores: print_str += f"Inst: {task_scores['inst_level_strict_acc']:.4f}  "
            print(print_str.strip())

            # Append to DF list
            for metric_name, score_value in task_scores.items():
                new_result = generate_result(
                    name=experiment_name,
                    total_params_str=total_params_str,
                    trainable_params_str=trainable_params_str,
                    task_name=task,
                    metric_name=metric_name,
                    score=score_value,
                    num_fewshot=num_fewshot
                )
                results_list.append(new_result)
                
    results_df = pd.DataFrame(results_list)
    return extracted_scores_dict, results_df, raw_results_dict