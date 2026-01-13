import os
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
import torch

from kdfp.schemas.suite import Suite
from kdfp.utils.general import get_timestamp

DEFAULT_CONFIG = "kdfp/configs/base.yaml"


def parse_cli_args():
    """Parse CLI arguments for the kdfp command"""
    parser = ArgumentParser(description="KDFP Command Line Interface")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Add 'run' command
    run_parser = subparsers.add_parser("run", help="Run the experiment pipeline")
    run_parser.add_argument(
        "-c",
        "--config",
        default=DEFAULT_CONFIG,
        help=f"Path to config file (default: {DEFAULT_CONFIG})",
    )

    return parser.parse_args()


def main(config_path: str):
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    torch.set_float32_matmul_precision("medium")

    suite = Suite.from_yaml(config_path)

    experiment_list, experiment_names = suite.generate_experiment_list()
    n = len(experiment_names)
    print(f"A total of {n} experiments have been specified:")
    for i, name in enumerate(experiment_names):
        assert experiment_list[i].cfg.name == name, (
            f"Experiment name mismatch: {experiment_list[i].cfg.name} != {name}"
        )
        print(f"    {i + 1}: {name}")

    results_list = []

    if suite.training_params.controls.get_baselines:
        print("\nBeginning baseline teacher testing...")
        results_list.extend(suite.run_baseline_teacher_tests())
        print("Baseline teacher testing complete!")

    print("\nStarting experiment runs...")
    for i, experiment in enumerate(experiment_list):
        print(f"\nStarting experiment {i + 1}/{n}: {experiment.cfg.name.upper()}")
        experiment.init()

        # Now that teachers have certainly been initialized, get benchmark scores.
        # We can get them from any experiment and only need to do it once, might as well
        # do the first one

        if i == 0 and suite.training_params.controls.get_baselines:
            print("\nBeginning teacher testing...")
            results_list.extend(
                experiment.test_teachers(
                    mode=suite.training_params.controls.testing_mode
                )
            )
            print("Teacher testing complete!\n")

        if suite.training_params.controls.run_or_test == "run":
            # this creates models, then tests them
            results_list.extend(experiment.run())
        else:
            # this tests existing models
            results_list.extend(
                experiment.test_student(
                    mode=suite.training_params.controls.testing_mode
                )
            )

        print(f"Finished experiment {i + 1}/{n}: {experiment.cfg.name.upper()}")

    base_scores_dir = Path(
        suite.parsed_dirs.root + "/" + suite.parsed_dirs.scores
    ).resolve()
    base_scores_dir.mkdir(parents=True, exist_ok=True)
    df_path = base_scores_dir / f"results_df_{get_timestamp()}.parquet"
    results_df = pd.DataFrame(results_list)
    results_df.to_parquet(df_path)
    results_df.to_markdown(base_scores_dir / f"results_df_{get_timestamp()}.md")
    print(
        f"Results dataframe for experiment configuration {config_path} saved in Parquet format to:\n\t{df_path}\n"
    )
    print(results_df)


def cli_main():
    """Entry point for the kdfp command"""
    args = parse_cli_args()

    if args.command == "run":
        main(config_path=args.config)
    else:
        print("Usage: kdfp run [--config CONFIG_FILE]")
        print("Run 'kdfp run --help' for more options")


if __name__ == "__main__":
    cli_main()
