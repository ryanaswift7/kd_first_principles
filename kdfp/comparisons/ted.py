import argparse
import gc
import json
import os
import shutil
import tempfile
from typing import Dict

# LM Eval imports
import lm_eval
import numpy as np

# Ray Imports
import ray
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from ray import train, tune
from ray.train import Checkpoint
from ray.tune.schedulers import ASHAScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm  # Import tqdm
from transformers import AutoTokenizer, GPTNeoXForCausalLM


# ==========================================
# 1. UTILITIES & ARCHITECTURE
# ==========================================
def is_valid_model_dir(path: str) -> bool:
    if not os.path.exists(path) or not os.path.isdir(path):
        return False
    files = set(os.listdir(path))
    has_config = "config.json" in files
    has_weights = any(f in files for f in ["model.safetensors", "pytorch_model.bin"])
    return has_config and has_weights


class TaskAwareFilter(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = output_dim

        # 2-Layer MLP with GeLU
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
        teacher_layers=32,
        student_layers=24,
        teacher_hidden=2560,
        student_hidden=2048,
        dtype=torch.float16,
    ):
        super().__init__()
        self.teacher = teacher
        self.student = student

        for param in self.teacher.parameters():
            param.requires_grad = False

        self.layer_map = np.linspace(0, teacher_layers - 1, student_layers, dtype=int)

        # Filters
        self.teacher_filters = nn.ModuleDict(
            {
                str(t_idx): TaskAwareFilter(teacher_hidden, teacher_hidden)
                for t_idx in self.layer_map
            }
        ).to(dtype=dtype)

        self.student_filters = nn.ModuleList(
            [
                TaskAwareFilter(student_hidden, teacher_hidden)
                for _ in range(student_layers)
            ]
        ).to(dtype=dtype)

    def compute_teacher_head_loss(self, hidden_state, input_ids):
        final_norm = self.teacher.gpt_neox.final_layer_norm
        lm_head = self.teacher.embed_out
        logits = lm_head(final_norm(hidden_state))

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = input_ids[..., 1:].contiguous()
        return F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )

    def forward(self, input_ids, attention_mask, stage="distill"):
        with torch.no_grad():
            t_out = self.teacher(
                input_ids, attention_mask=attention_mask, output_hidden_states=True
            )
            t_hiddens = t_out.hidden_states[1:]
            t_logits = t_out.logits

        if stage == "teacher_warmup":
            s_hiddens, s_logits = None, None
        else:
            s_out = self.student(
                input_ids, attention_mask=attention_mask, output_hidden_states=True
            )
            s_logits, s_hiddens = s_out.logits, s_out.hidden_states[1:]

        # Stage 1: Teacher Filter Warmup
        if stage == "teacher_warmup":
            loss = 0.0
            for t_idx in self.layer_map:
                t_filt = self.teacher_filters[str(t_idx)](t_hiddens[t_idx])
                loss += self.compute_teacher_head_loss(t_filt, input_ids)
            return loss / len(self.layer_map)

        # Stage 2: Student Filter Warmup
        elif stage == "student_warmup":
            loss = 0.0
            for s_idx in range(len(self.student_filters)):
                s_filt = self.student_filters[s_idx](s_hiddens[s_idx])
                loss += self.compute_teacher_head_loss(s_filt, input_ids)
            return loss / len(self.student_filters)

        # Stage 3: Distillation (MSE + CE + KL)
        elif stage == "distill":
            # A. Layer MSE
            layer_loss = 0.0
            mse = nn.MSELoss()
            for s_idx, t_idx in enumerate(self.layer_map):
                with torch.no_grad():
                    t_target = self.teacher_filters[str(t_idx)](t_hiddens[t_idx])
                s_mapped = self.student_filters[s_idx](s_hiddens[s_idx])
                layer_loss += mse(s_mapped, t_target)
            layer_loss = layer_loss / len(self.layer_map)

            # B. Task CE
            shift_logits = s_logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            ce_loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
            )

            # C. KL Divergence
            T = 2.0
            kl_loss = F.kl_div(
                F.log_softmax(s_logits / T, dim=-1),
                F.softmax(t_logits / T, dim=-1),
                reduction="batchmean",
            ) * (T**2)

            return ce_loss + (0.1 * layer_loss) + (1.0 * kl_loss)


# ==========================================
# 2. RAY TUNE TRAINING FUNCTION (STREAMING)
# ==========================================
def train_ted_ray(config: Dict):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = (
        torch.bfloat16
        if (torch.cuda.is_available() and torch.cuda.is_bf16_supported())
        else torch.float16
    )

    # 1. Models
    teacher = GPTNeoXForCausalLM.from_pretrained(
        "EleutherAI/pythia-2.8b", torch_dtype=dtype, device_map="auto"
    )
    student = GPTNeoXForCausalLM.from_pretrained(
        "EleutherAI/pythia-1.4b", torch_dtype=dtype
    ).to(device)
    student.gradient_checkpointing_enable()

    model = TEDDistiller(teacher, student, dtype=dtype).to(device)

    # 2. Streaming Data Setup
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-1.4b")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Use streaming=True
    dataset = load_dataset(
        "HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True
    )

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
    # Single iterator for the entire lifecycle ensures we move linearly through dataset
    data_iterator = iter(loader)

    # 3. Training Loop Helper (with Progress Bar)
    def run_stage_for_tokens(stage_name, target_tokens, optimizer):
        tokens_processed = 0
        last_loss = 0.0

        # Initialize Progress Bar
        pbar = tqdm(total=target_tokens, desc=f"[{stage_name}]", unit="tok", leave=True)

        while tokens_processed < target_tokens:
            try:
                batch = next(data_iterator)
            except StopIteration:
                print(f"WARNING: Dataset exhausted during {stage_name}!")
                break

            input_ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            current_tokens = input_ids.numel()

            with torch.amp.autocast(device_type="cuda", dtype=dtype):
                loss = model(input_ids, mask, stage_name)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            tokens_processed += current_tokens
            last_loss = loss.item()

            # Update Progress Bar
            pbar.update(current_tokens)
            pbar.set_postfix({"loss": f"{last_loss:.4f}"})

        pbar.close()
        return last_loss

    # --- STAGE 1: Teacher Warmup ---
    # We use data 0 -> X
    opt_t = torch.optim.AdamW(
        model.teacher_filters.parameters(), lr=config["lr_filters"]
    )
    model.train()
    run_stage_for_tokens("teacher_warmup", config["tokens_s1"], opt_t)

    # --- STAGE 2: Student Warmup ---
    # We use data X -> Y
    opt_s = torch.optim.AdamW(
        model.student_filters.parameters(), lr=config["lr_filters"]
    )
    run_stage_for_tokens("student_warmup", config["tokens_s2"], opt_s)

    # --- STAGE 3: Distillation ---
    # We use data Y -> Z
    params = list(model.student.parameters()) + list(model.student_filters.parameters())
    opt_d = torch.optim.AdamW(params, lr=config["lr_student"])

    # Pruning reports every 25% of the distill budget
    chunk_size = config["tokens_s3"] // 4
    cumulative_tokens = 0  # Tracker for ASHA

    for i in range(4):
        loss_val = run_stage_for_tokens("distill", chunk_size, opt_d)
        cumulative_tokens += chunk_size

        with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
            model.student.save_pretrained(temp_checkpoint_dir)
            tokenizer.save_pretrained(temp_checkpoint_dir)

            train.report(
                {
                    "loss": loss_val,
                    "tokens_processed": cumulative_tokens,  # Report explicit metric
                },
                checkpoint=Checkpoint.from_directory(temp_checkpoint_dir),
            )


# ==========================================
# 3. EVALUATION
# ==========================================
def run_evaluation(model_path, tasks):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n=== Running Eval on {model_path} ===")

    if not is_valid_model_dir(model_path):
        print(f"Error: No valid model found at {model_path}")
        return

    # Dictionary to store only the final scores
    scores_only = {}

    for shot_count in [0, 5]:
        print(f"\n--- Running {shot_count}-Shot Evaluation ---")
        results = lm_eval.simple_evaluate(
            model="hf",
            model_args=f"pretrained={model_path},dtype=auto",
            tasks=tasks,
            num_fewshot=shot_count,
            batch_size="auto",
            device=device,
        )

        shot_key = f"{shot_count}_shot"
        scores_only[shot_key] = {}

        print(f"Results for {shot_count}-Shot:")
        if "results" in results:
            for task, metrics in results["results"].items():
                acc = metrics.get("acc,none", metrics.get("acc", 0.0))
                norm = metrics.get("acc_norm,none", metrics.get("acc_norm", 0.0))

                # Save only the essential metrics to our dictionary
                scores_only[shot_key][task] = {"acc": acc, "acc_norm": norm}

                print(f"{task:<20} | Acc: {acc:.4f} | Norm: {norm:.4f}")

    # Save only the extracted scores
    out_file = os.path.join(model_path, "eval_scores.json")
    with open(out_file, "w") as f:
        json.dump(scores_only, f, indent=4)

    print(f"\nScores saved to {out_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--save_path", type=str, required=True)

    # Token Budgets
    parser.add_argument(
        "--tokens_s1",
        type=int,
        default=10_000_000,
        help="Tokens for Stage 1 (Teacher Warmup)",
    )
    parser.add_argument(
        "--tokens_s2",
        type=int,
        default=10_000_000,
        help="Tokens for Stage 2 (Student Warmup)",
    )
    parser.add_argument(
        "--tokens_s3",
        type=int,
        default=100_000_000,
        help="Tokens for Stage 3 (Distillation)",
    )
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seq_len", type=int, default=2048)

    args = parser.parse_args()

    eval_tasks = [
        "hellaswag",
        "arc_easy",
        "arc_challenge",
        "winogrande",
        "boolq",
        "piqa",
        "sciq",
    ]

    should_train = True
    if args.skip_train:
        should_train = False
    elif is_valid_model_dir(args.save_path):
        print(f"Model found at {args.save_path}. Skipping training.")
        should_train = False

    if should_train:
        print("=== Starting Ray Tune Sweep (Streaming) ===")
        print(
            f"S1 Tokens: {args.tokens_s1:_}, S2 Tokens: {args.tokens_s2:_}, S3 Tokens: {args.tokens_s3:_}"
        )

        ray.init()

        search_space = {
            "lr_filters": tune.loguniform(1e-4, 1e-2),
            "lr_student": tune.loguniform(1e-6, 1e-4),
            "tokens_s1": args.tokens_s1,
            "tokens_s2": args.tokens_s2,
            "tokens_s3": args.tokens_s3,
            "batch_size": args.batch_size,
            "seq_len": args.seq_len,
        }

        # ASHA Scheduler configured for TOKENS
        # time_attr: tells ASHA to look at 'tokens_processed' metric in report()
        # max_t: max tokens for the run
        # grace_period: minimum tokens before pruning (1 chunk / 25%)
        scheduler = ASHAScheduler(
            time_attr="tokens_processed",
            max_t=args.tokens_s3,
            grace_period=args.tokens_s3 // 4,
            reduction_factor=2,
        )

        tuner = tune.Tuner(
            tune.with_resources(train_ted_ray, resources={"cpu": 4, "gpu": 1}),
            param_space=search_space,
            tune_config=tune.TuneConfig(
                metric="loss",
                mode="min",
                scheduler=scheduler,
                num_samples=4,
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

        ray.shutdown()
        torch.cuda.empty_cache()
        gc.collect()

    run_evaluation(args.save_path, eval_tasks)


# ==========================================
# 4. MAIN
# ==========================================
if __name__ == "__main__":
    main()
