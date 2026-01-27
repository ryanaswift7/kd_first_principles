from datasets import Dataset, DatasetDict, load_dataset
from tqdm import tqdm
from transformers import AutoTokenizer

REPO_ID = "ryanaswift7/fineweb-edu-pythia-kdfp-subsets"
TOKENIZER_ID = "EleutherAI/pythia-70m"

TRAIN_TARGET = 10_000_000
VAL_TARGET = 1_000_000
TEST_TARGET = 1_000_000

MASTER_SEED = 42
BUFFER_SIZE = 10_000
VERSIONS = ["v1", "v2", "v3", "v4"]


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
    print(f"--- Initializing Single Continuous Stream (Master Seed: {MASTER_SEED}) ---")
    print(
        f"Targets: Train={TRAIN_TARGET / 1e6}M | Val={VAL_TARGET / 1e6}M | Test={TEST_TARGET / 1e6}M"
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
            ds_iterator, tokenizer, TRAIN_TARGET, version, "train"
        )
        val_data = get_next_chunk(
            ds_iterator, tokenizer, VAL_TARGET, version, "validation"
        )
        test_data = get_next_chunk(ds_iterator, tokenizer, TEST_TARGET, version, "test")

        train_ds = Dataset.from_list(train_data)
        val_ds = Dataset.from_list(val_data)
        test_ds = Dataset.from_list(test_data)

        final_dataset_dict = DatasetDict(
            {"train": train_ds, "validation": val_ds, "test": test_ds}
        )

        print(f"  > Train: {len(train_ds)} docs")
        print(f"  > Val:   {len(val_ds)} docs")
        print(f"  > Test:  {len(test_ds)} docs")

        print(f"  > Uploading {version} to {REPO_ID}...")
        final_dataset_dict.push_to_hub(REPO_ID, config_name=version)
        print(f"  ✅ {version} uploaded successfully.")

    print("\n------------------------------------------------")
    print("All versions uploaded!")
    print(f"Check your subsets at: https://huggingface.co/{REPO_ID}")


if __name__ == "__main__":
    main()
