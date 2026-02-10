import argparse
import sys

from datasets import Dataset, DatasetDict, load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

# Base name for the repository. The size will be appended to this (e.g., "-10m")
BASE_REPO_ID = "ryanaswift7/fineweb-edu-pythia-kdfp-subsets"
TOKENIZER_ID = "EleutherAI/pythia-70m"

MASTER_SEED = 3
BUFFER_SIZE = 10_000
VERSIONS = ["v1", "v2", "v3", "v4"]
MAX_TOKENS = 2048  # Filter threshold


def parse_size(size_str):
    """Parses a string like '10k', '100m', '2b' into an integer."""
    size_str = size_str.lower().strip()
    multipliers = {"k": 1_000, "m": 1_000_000, "b": 1_000_000_000}

    if size_str[-1] in multipliers:
        try:
            number = float(size_str[:-1])
            return int(number * multipliers[size_str[-1]])
        except ValueError:
            pass

    try:
        return int(size_str)
    except ValueError:
        raise ValueError(
            f"Invalid size format: {size_str}. Use suffix k, m, or b (e.g., 100m)."
        )


def get_next_chunk(iterator, tokenizer, target_tokens, version_label, split_name):
    buffer = []
    current_count = 0
    desc = f"Collecting {version_label} [{split_name}]"
    pbar = tqdm(total=target_tokens, desc=desc, unit="tok")

    while current_count < target_tokens:
        try:
            sample = next(iterator)
            text = sample.get("text", "")

            tokens = tokenizer(text, add_special_tokens=False)["input_ids"]
            num_tokens = len(tokens)

            # --- FILTER LOGIC START ---
            if num_tokens > MAX_TOKENS:
                continue  # Skip this sample and move to the next
            # --- FILTER LOGIC END ---

            buffer.append(
                {
                    "text": text,
                    "token_count": num_tokens,
                    "id": sample.get("id"),
                    "url": sample.get("url"),
                    "edu_score": sample.get("score"),
                    "version_label": version_label,
                    "split_name": split_name,
                }
            )

            current_count += num_tokens
            pbar.update(num_tokens)

        except StopIteration:
            print(
                f"\nCRITICAL: Dataset ran out of data during {version_label} ({split_name})!"
            )
            break

    pbar.close()
    return buffer


def main():
    parser = argparse.ArgumentParser(
        description="Upload Fineweb-Edu subsets based on token count."
    )
    parser.add_argument(
        "train_size",
        type=str,
        help="Number of tokens for the train split (e.g., 10k, 100m, 2b).",
    )

    args = parser.parse_args()

    try:
        train_target = parse_size(args.train_size)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    val_target = int(train_target * 0.10)
    test_target = int(train_target * 0.10)

    size_suffix = args.train_size.lower().strip()
    final_repo_id = f"{BASE_REPO_ID}-{size_suffix}"

    print(f"--- Initializing Single Continuous Stream (Master Seed: {MASTER_SEED}) ---")
    print(f"Input Argument: {args.train_size}")
    print(f"Target Repo:  {final_repo_id}")
    print(f"Max Seq Len:  {MAX_TOKENS}")
    print(
        f"Targets: Train={train_target:,} | Val={val_target:,} | Test={test_target:,}"
    )

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_ID)
    ds_stream = load_dataset(
        "HuggingFaceFW/fineweb-edu", name="sample-10BT", split="train", streaming=True
    )

    shuffled_stream = ds_stream.shuffle(seed=MASTER_SEED, buffer_size=BUFFER_SIZE)
    ds_iterator = iter(shuffled_stream)

    for version in VERSIONS:
        print(f"\n=== Processing {version} ===")

        train_data = get_next_chunk(
            ds_iterator, tokenizer, train_target, version, "train"
        )
        val_data = get_next_chunk(
            ds_iterator, tokenizer, val_target, version, "validation"
        )
        test_data = get_next_chunk(ds_iterator, tokenizer, test_target, version, "test")

        train_ds = Dataset.from_list(train_data)
        val_ds = Dataset.from_list(val_data)
        test_ds = Dataset.from_list(test_data)

        final_dataset_dict = DatasetDict(
            {"train": train_ds, "validation": val_ds, "test": test_ds}
        )

        print(f"  > Train: {len(train_ds)} docs")
        print(f"  > Val:   {len(val_ds)} docs")
        print(f"  > Test:  {len(test_ds)} docs")

        print(f"  > Uploading {version} to {final_repo_id}...")
        final_dataset_dict.push_to_hub(final_repo_id, config_name=version)
        print(f"  ✅ {version} uploaded successfully.")

    print("\n------------------------------------------------")
    print("All versions uploaded!")
    print(f"Check your subsets at: https://huggingface.co/{final_repo_id}")


if __name__ == "__main__":
    main()
