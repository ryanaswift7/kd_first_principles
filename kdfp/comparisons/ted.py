import argparse
import gc
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict

import numpy as np
import ray
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from datasets import load_dataset
from ray import train, tune
from ray.train import Checkpoint
from ray.tune.schedulers import ASHAScheduler
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from kdfp.utils.evals import evaluate_model
from kdfp.utils.general import is_valid_model_dir


EVAL_TASKS = [
    "arc_challenge",
    "arc_easy",
    "boolq",
    "triviaqa",
    "piqa",
    "sciq",
    "winogrande",
    "mmlu",
    "drop",
]

MODEL_CONFIGS = {
    "pythia": {
        "teacher": "EleutherAI/pythia-2.8b-deduped",
        "student": "EleutherAI/pythia-1.4b-deduped",
        "teacher_layers": 32,
        "student_layers": 24,
        "teacher_embedding_dim": 2560,
        "student_embedding_dim": 2048,
    },
    "llama3.2": {
        "teacher": "meta-llama/Llama-3.2-3B",
        "student": "meta-llama/Llama-3.2-1B",
        "teacher_layers": 28,
        "student_layers": 16,
        "teacher_embedding_dim": 3072,
        "student_embedding_dim": 2048,
    },
    "qwen2.5": {
        "teacher": "Qwen/Qwen2.5-3B",
        "student": "Qwen/Qwen2.5-1.5B",
        "teacher_layers": 36,
        "student_layers": 28,
        "teacher_embedding_dim": 2048,
        "student_embedding_dim": 1536,
    },
    "gemma3": {
        "teacher": "google/gemma-3-4b-pt",
        "student": "google/gemma-3-1b-pt",
        "teacher_layers": 34,
        "student_layers": 26,
        "teacher_embedding_dim": 2560,
        "student_embedding_dim": 2304,
    },
    "qwen3": {
        "teacher": "Qwen/Qwen3-4B",
        "student": "Qwen/Qwen3-1.7B",
        "teacher_layers": 36,
        "student_layers": 28,
        "teacher_embedding_dim": 2560,
        "student_embedding_dim": 2048,
    }
}

class TaskAwareFilter(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = output_dim

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=False),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim, bias=False),
        )

    def forward(self, hidden_state):
        return self.mlp(hidden_state)


class TEDDistiller(nn.Module):
    def __init__(
        self,
        teacher,
        student,
        teacher_layers,
        student_layers,
        teacher_embedding_dim,
        student_embedding_dim,
        dtype=torch.bfloat16,
    ):
        super().__init__()
        self.teacher = teacher
        self.student = student

        # Freeze Teacher
        for param in self.teacher.parameters():
            param.requires_grad = False
        self.teacher.eval() # Ensure dropout layers are disabled

        self.layer_map = np.linspace(0, teacher_layers - 1, student_layers, dtype=int)

        self.teacher_filters = nn.ModuleDict(
            {
                str(t_idx): TaskAwareFilter(teacher_embedding_dim, teacher_embedding_dim)
                for t_idx in self.layer_map
            }
        ).to(dtype=dtype)

        self.student_filters = nn.ModuleList(
            [
                TaskAwareFilter(student_embedding_dim, teacher_embedding_dim)
                for _ in range(student_layers)
            ]
        ).to(dtype=dtype)

    def get_norm_and_head(self, model):
        """Dynamically extract final layernorm and unembed head based on architecture."""
        if hasattr(model, "model") and hasattr(model.model, "norm"):
            return model.model.norm, model.lm_head
        elif hasattr(model, "gpt_neox"):
            return model.gpt_neox.final_layer_norm, model.embed_out
        else:
            raise ValueError(f"Unsupported model architecture for extracting norm/head: {type(model)}")

    def compute_teacher_head_loss(self, hidden_state, input_ids):
        norm, head = self.get_norm_and_head(self.teacher)
        
        normed_state = norm(hidden_state)
        logits = head(normed_state)

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        
        return F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )

    def forward(self, input_ids, stage="distill"):
        with torch.no_grad():
            t_outputs = self.teacher(
                input_ids, 
                output_hidden_states=True,
                return_dict=True,
                use_cache=False # VRAM FIX: Stop KV cache allocation during training
            )
            t_logits = t_outputs.logits
            t_hiddens = {int(l): t_outputs.hidden_states[int(l) + 1] for l in self.layer_map}
            del t_outputs 

        if stage == "teacher_warmup":
            s_hiddens, s_logits, s_loss = None, None, None
        else:
            s_outputs = self.student(
                input_ids, 
                labels=input_ids,
                output_hidden_states=True,
                return_dict=True,
                use_cache=False # VRAM FIX: Stop KV cache allocation during training
            )
            s_logits = s_outputs.logits
            s_loss = s_outputs.loss
            s_num_layers = self.student.config.num_hidden_layers
            
            s_hiddens = {l: s_outputs.hidden_states[l + 1] for l in range(s_num_layers)}
            del s_outputs

        if stage == "teacher_warmup":
            loss = 0.0
            for t_idx in self.layer_map:
                t_filt = self.teacher_filters[str(t_idx)](t_hiddens.pop(int(t_idx)))
                loss += self.compute_teacher_head_loss(t_filt, input_ids)
            return loss / len(self.layer_map)

        elif stage == "student_warmup":
            loss = 0.0
            for s_idx in range(len(self.student_filters)):
                s_filt = self.student_filters[s_idx](s_hiddens.pop(s_idx))
                loss += self.compute_teacher_head_loss(s_filt, input_ids)
            return loss / len(self.student_filters)

        elif stage == "distill":
            # --- VRAM FIX: Calculate KL Divergence FIRST ---
            # This allows us to delete the massive [batch, seq, vocab] logits early
            ce_loss = s_loss
            T = 2.0
            kl_loss = F.kl_div(
                F.log_softmax(s_logits / T, dim=-1),
                F.softmax(t_logits / T, dim=-1),
                reduction="batchmean",
            ) * (T**2)

            del t_logits, s_logits 
            # --- END VRAM FIX ---

            # Now that big memory is clear, compute layer MSE loss
            layer_loss = 0.0
            mse = nn.MSELoss()
            for s_idx, t_idx in enumerate(self.layer_map):
                with torch.no_grad():
                    t_target = self.teacher_filters[str(t_idx)](t_hiddens.pop(int(t_idx)))
                s_mapped = self.student_filters[s_idx](s_hiddens.pop(s_idx))
                layer_loss += mse(s_mapped, t_target)
            layer_loss = layer_loss / len(self.layer_map)

            return ce_loss + (0.1 * layer_loss) + (2.5 * kl_loss)
                

def train_ted_ray(config: Dict):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16

    model_family = config["model_family"]
    model_cfg = MODEL_CONFIGS[model_family]

    print(f"Loading Teacher: {model_cfg['teacher']} from Hugging Face...")
    teacher = AutoModelForCausalLM.from_pretrained(
        model_cfg["teacher"], 
        torch_dtype=dtype, 
        attn_implementation="sdpa", # VRAM FIX: Native memory-efficient attention
    ).to(device)
    
    print(f"Loading Student: {model_cfg['student']} from Hugging Face...")
    student = AutoModelForCausalLM.from_pretrained(
        model_cfg["student"], 
        torch_dtype=dtype, 
        attn_implementation="sdpa", # VRAM FIX: Native memory-efficient attention
    ).to(device)
    
    # VRAM FIX: Enable Gradient Checkpointing
    student.gradient_checkpointing_enable() 
    
    model = TEDDistiller(
        teacher, 
        student, 
        teacher_layers=model_cfg["teacher_layers"],
        student_layers=model_cfg["student_layers"],
        teacher_embedding_dim=model_cfg["teacher_embedding_dim"],
        student_embedding_dim=model_cfg["student_embedding_dim"],
        dtype=dtype
    ).to(device)

    # 2. Streaming Data Setup
    tokenizer = AutoTokenizer.from_pretrained(model_cfg["student"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset_name = config["dataset"]
    if dataset_name == "fineweb-edu":
        dataset = load_dataset("ryanaswift7/fineweb-edu-pythia-kdfp-subsets-100m", "v1", split="train", streaming=True)
    elif dataset_name == "wikipedia":
        dataset = load_dataset("ryanaswift7/wikipedia-pythia-kdfp-subsets-100m", "v1", split="train", streaming=True)
    elif dataset_name == "dolma3":
        dataset = load_dataset("ryanaswift7/dolma3-pythia-kdfp-subsets-100m", "v1", split="train", streaming=True)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    def tokenize(ex):
        return tokenizer(
            ex["text"],
            truncation=True,
            padding="max_length",
            max_length=config["seq_len"],
        )

    dataset = dataset.map(tokenize, batched=True, remove_columns=["text"])
    dataset = dataset.with_format("torch")

    loader = DataLoader(dataset, batch_size=config["batch_size"])
    data_iterator = iter(loader)

    # 3. Training Loop Helper
    def run_stage_for_tokens(stage_name, target_tokens, optimizer):
        tokens_processed = 0
        last_loss = 0.0
        step_count = 0

        pbar = tqdm(total=target_tokens, desc=f"[{stage_name}]", unit="tok", leave=True)

        while tokens_processed < target_tokens:
            try:
                batch = next(data_iterator)
            except StopIteration:
                print(f"WARNING: Dataset exhausted during {stage_name}!")
                break

            input_ids = batch["input_ids"].to(device)
            current_tokens = input_ids.numel()

            loss = model(input_ids, stage_name)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            tokens_processed += current_tokens
            last_loss = loss.item()

            pbar.update(current_tokens)
            pbar.set_postfix({"loss": f"{last_loss:.4f}"})
            
            del loss, input_ids, batch
            
            step_count += 1
            if step_count % 50 == 0:
                gc.collect()
                torch.cuda.empty_cache()

        pbar.close()
        return last_loss

    # VRAM FIX: Fused AdamW avoids creating intermediate copies of gradients
    opt_t = torch.optim.AdamW(model.teacher_filters.parameters(), lr=config["lr_filters"], fused=True)
    model.train()
    run_stage_for_tokens("teacher_warmup", config["tokens_s1"], opt_t)

    opt_s = torch.optim.AdamW(model.student_filters.parameters(), lr=config["lr_filters"], fused=True)
    run_stage_for_tokens("student_warmup", config["tokens_s2"], opt_s)

    params = list(model.student.parameters()) + list(model.student_filters.parameters())
    opt_d = torch.optim.AdamW(params, lr=config["lr_student"], fused=True)

    chunk_size = config["tokens_s3"] // 4
    cumulative_tokens = 0  

    for i in range(4):
        loss_val = run_stage_for_tokens("distill", chunk_size, opt_d)
        cumulative_tokens += chunk_size

        with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
            model.student.save_pretrained(temp_checkpoint_dir)
            tokenizer.save_pretrained(temp_checkpoint_dir)

            train.report(
                {
                    "loss": loss_val,
                    "tokens_processed": cumulative_tokens,
                },
                checkpoint=Checkpoint.from_directory(temp_checkpoint_dir),
            )
        gc.collect()
        torch.cuda.empty_cache()
            
    del model
    del teacher
    del student
    del opt_t
    del opt_s
    del opt_d
    
    gc.collect()
    torch.cuda.empty_cache()

def run_evaluation(model_path: str | Path, tasks: list, eval_limit: int = None, device: str = "cuda"):
    model_path = Path(model_path)
    print(f"\n=== Running Eval on {model_path} ===")

    if not is_valid_model_dir(model_path):
        print(f"Error: No valid model found at {model_path}")
        return

    experiment_name = model_path.name 
    master_results_df = pd.DataFrame()
    master_raw_dict = {}
    master_extracted_dict = {}

    for n in [0]:
        print(f"\n--- Benchmarking all tasks with {n} few-shots ---")
        extracted_dict, results_df, raw_results_dict = evaluate_model(
            model_path=str(model_path), 
            tasks=tasks,
            experiment_name=experiment_name,
            num_fewshot=n,
            batch_size="auto",
            limit=eval_limit,
            device=device
        )

        fewshot_key = f"fewshot_{n}"
        master_raw_dict[fewshot_key] = raw_results_dict
        master_extracted_dict[fewshot_key] = extracted_dict

        if not results_df.empty:
            if master_results_df.empty:
                master_results_df = results_df
            else:
                master_results_df = pd.concat([master_results_df, results_df], ignore_index=True)

    out_dir = model_path
    
    with open(out_dir / "raw_eval_scores.json", "w") as f:
        json.dump(master_raw_dict, f, indent=4)
        
    with open(out_dir / "extracted_eval_scores.json", "w") as f:
        json.dump(master_extracted_dict, f, indent=4)

    if not master_results_df.empty:
        master_results_df.to_csv(out_dir / "eval_scores.csv", index=False)
        master_results_df.to_parquet(out_dir / "eval_scores.parquet", index=False)
        print(f"\nScores successfully saved to {out_dir}")
    else:
        print("\nNo valid metrics were found to save.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--save-path", type=str, required=True)
    parser.add_argument("--model-family", type=str, choices=["pythia", "llama3.2", "qwen2.5", "gemma3", "qwen3"], required=True)
    parser.add_argument("--dataset", type=str, choices=["fineweb-edu", "wikipedia", "dolma3"], required=True)
    
    parser.add_argument("--lr-filters", type=float, default=None, 
                        help="Specify a fixed learning rate for the filters.")
    parser.add_argument("--lr-student", type=float, default=None, 
                        help="Specify a fixed learning rate for the student.")
    
    parser.add_argument("--force-lr-tune", action="store_true", 
                        help="Force Ray Tune to run multiple trials even if exact learning rates are provided.")

    parser.add_argument("--tokens-s1", type=int, default=10_000_000)
    parser.add_argument("--tokens-s2", type=int, default=10_000_000)
    parser.add_argument("--tokens-s3", type=int, default=100_000_000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--eval-limit", type=int, default=None)

    args = parser.parse_args()

    should_train = True
    if args.skip_train:
        should_train = False
    elif is_valid_model_dir(args.save_path):
        print(f"Model found at {args.save_path}. Skipping training.")
        should_train = False

    if should_train:
        print("=== Starting Ray Tune Sweep (Streaming) ===")
        print(f"Model Family: {args.model_family} | Dataset: {args.dataset}")

        ray.init()

        search_space = {
            "lr_filters": args.lr_filters if args.lr_filters is not None else tune.qloguniform(1e-4, 1e-2, 1e-5),
            "lr_student": args.lr_student if args.lr_student is not None else tune.qloguniform(1e-6, 1e-4, 1e-7),
            "tokens_s1": args.tokens_s1,
            "tokens_s2": args.tokens_s2,
            "tokens_s3": args.tokens_s3,
            "batch_size": args.batch_size,
            "seq_len": args.seq_len,
            "model_family": args.model_family,
            "dataset": args.dataset,
        }

        scheduler = ASHAScheduler(
            time_attr="tokens_processed",
            max_t=args.tokens_s3,
            grace_period=args.tokens_s3 // 4,
            reduction_factor=2,
        )

        num_trials = 5
        if args.lr_filters is not None and args.lr_student is not None and not args.force_lr_tune:
            num_trials = 1
            print("Explicit learning rates provided. Running a single training trial instead of a sweep.")
        elif args.force_lr_tune and (args.lr_filters is not None and args.lr_student is not None):
            print("WARNING: --force-lr-tune active with explicit learning rates. Running 10 trials with identical hyperparameters to test variance.")

        tuner = tune.Tuner(
            tune.with_resources(train_ted_ray, resources={"cpu": 4, "gpu": 1}),
            param_space=search_space,
            tune_config=tune.TuneConfig(
                metric="loss",
                mode="min",
                scheduler=scheduler,
                num_samples=num_trials,
            ),
            run_config=train.RunConfig(
                storage_path=os.path.abspath("./ray_results"), name="ted_stream_sweep"
            ),
        )

        results = tuner.fit()
        best_result = results.get_best_result("loss", "min")

        print(f"Best Trial Loss: {best_result.metrics['loss']}")
        print(f"Restoring best model to {args.save_path}...")

        with best_result.checkpoint.as_directory() as checkpoint_dir:
            shutil.copytree(checkpoint_dir, args.save_path, dirs_exist_ok=True)

        best_config = best_result.config
        final_lrs = {
            "lr_filters": best_config.get("lr_filters"),
            "lr_student": best_config.get("lr_student")
        }
        
        lr_save_path = Path(args.save_path) / "final_learning_rates.json"
        with open(lr_save_path, "w") as f:
            json.dump(final_lrs, f, indent=4)
        print(f"Saved optimal learning rates to {lr_save_path}")

        ray.shutdown()
        torch.cuda.empty_cache()
        gc.collect()

    run_evaluation(args.save_path, EVAL_TASKS, args.eval_limit)

if __name__ == "__main__":
    main()