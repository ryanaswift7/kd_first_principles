import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl.experimental.minillm import MiniLLMConfig, MiniLLMTrainer

# --- Configuration ---
STUDENT_MODEL_ID = "EleutherAI/pythia-1.4b-deduped"
TEACHER_MODEL_ID = "EleutherAI/pythia-2.8b-deduped"
DATASET_ID = "ryanaswift7/fineweb-edu-pythia-kdfp-subsets-100m"
DATASET_CONFIG = "v1"
OUTPUT_DIR = "results/minillm-pythia-1.4b-100m"

# MiniLLM / Training Hyperparameters
MAX_LENGTH = 2048
PROMPT_LENGTH = 128


def main():
    print("--- Starting Official MiniLLM Training ---")

    # 1. Load Tokenizer
    # MiniLLM requires the tokenizer to be passed clearly for decoding/encoding during generation
    tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    # tokenizer.padding_side = "left"  # Crucial for generation-based training

    # 2. Load and Format Dataset
    # MiniLLM is an RL/Generation method (Reverse KLD).
    # It needs a "prompt" to generate a response from the student to compare with the teacher.
    # Since FineWeb-Edu is raw text, we split the document: First N tokens = Prompt.
    print(f"Loading dataset: {DATASET_ID}")
    dataset = load_dataset(DATASET_ID, DATASET_CONFIG, split="train")

    def format_for_minillm(example):
        # We take the raw text, tokenize it, and split it.
        # The Trainer expects 'input_ids' (prompt) and usually 'labels' (target)
        # but MiniLLMTrainer is unique. It generally looks for a 'prompt' column to initiate generation.

        full_tokens = tokenizer(
            example["text"], truncation=True, max_length=MAX_LENGTH
        )["input_ids"]

        if len(full_tokens) <= PROMPT_LENGTH + 10:
            return None  # Skip short docs

        # Create prompt text
        prompt_tokens = full_tokens[:PROMPT_LENGTH]
        prompt_text = tokenizer.decode(prompt_tokens)

        # We return the prompt. The trainer's data collator handles tokenization of the prompt
        return {"prompt": prompt_text}

    print("Formatting dataset (Splitting Prompts)...")
    dataset = dataset.map(format_for_minillm, remove_columns=dataset.column_names)
    dataset = dataset.filter(lambda x: x is not None)  # Remove skipped items
    print(f"Training on {len(dataset)} documents.")

    # 3. Load Models
    print("Loading Teacher (2.8B)...")
    teacher_model = AutoModelForCausalLM.from_pretrained(
        TEACHER_MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )
    # Teacher must be frozen
    teacher_model.eval()
    teacher_model.requires_grad_(False)

    print("Loading Student (1.4B)...")
    student_model = AutoModelForCausalLM.from_pretrained(
        STUDENT_MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="flash_attention_2",
    )

    # 4. Configure MiniLLM
    # MiniLLMConfig inherits from TrainingArguments, so ALL args go here.
    minillm_config = MiniLLMConfig(
        model_init_kwargs=None,  # We loaded models manually
        # --- Standard Training Args ---
        output_dir=OUTPUT_DIR,
        learning_rate=5e-6,  # MiniLLM uses low LR (PPO-style)
        num_train_epochs=1,
        per_device_train_batch_size=1,  # Generation is memory heavy
        gradient_accumulation_steps=8,
        bf16=True,  # Essential for Ampere+
        logging_steps=100,
        save_strategy="epoch",
        report_to="none",
        gradient_checkpointing=True,
        ddp_find_unused_parameters=False,
        # --- MiniLLM Specific Args ---
        # "model_name" usually required by TRL configs for tagging
        model_name=STUDENT_MODEL_ID,
        # Generation Constraints
        max_length=MAX_LENGTH,  # Total sequence length
        max_prompt_length=PROMPT_LENGTH,  # Length of prompt part
        # Algorithm Hyperparameters (from the paper)
        beta=0.5,  # KL penalty coefficient
        rkl_advantage=True,  # Use Reverse KL Advantage (Core feature)
        single_step_decomposition=True,  # Optimization trick from paper
        length_normalization=True,  # Normalize reward by length
    )

    # 5. Initialize Trainer
    trainer = MiniLLMTrainer(
        model=student_model,
        teacher_model=teacher_model,
        train_dataset=dataset,
        tokenizer=tokenizer,
        args=minillm_config,
    )

    # 6. Train
    print("Starting Training...")
    trainer.train()

    # 7. Save
    print("Saving model...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)


if __name__ == "__main__":
    main()
