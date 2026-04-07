import os
from argparse import ArgumentParser
from typing import Optional
import gc
import pandas as pd
import torch

from kdfp.schemas.suite import Suite
from kdfp.utils.general import get_timestamp

DEFAULT_CONFIG = "kdfp/configs/base_100m_tokens.yaml"


def run_suite(suite_config_path: str):
    """
    The core logic of the pipeline.
    This can be called safely from other Python scripts.
    """
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    torch.set_float32_matmul_precision("medium")

    suite = Suite.from_yaml(suite_config_path)

    experiment_list, experiment_names = suite.generate_experiment_list()
    n = len(experiment_names)
    print(f"A total of {n} experiments have been specified:")
    for i, name in enumerate(experiment_names):
        assert experiment_list[i].cfg.name == name, (
            f"Experiment name mismatch: {experiment_list[i].cfg.name} != {name}"
        )
        print(f"    {i + 1}: {name}")

    results_list = []

    # if suite.training_params.controls.get_baselines:
    #     print("\nBeginning baseline teacher testing...")
    #     results_list.extend(suite.run_baseline_teacher_tests())
    #     print("Baseline teacher testing complete!")

    print("\nStarting experiment runs...")
    for i, experiment in enumerate(experiment_list):
        print(f"\nStarting experiment {i + 1}/{n}: {experiment.cfg.name.upper()}")
        experiment.init()

        # if i == 0 and suite.training_params.controls.get_baselines:
        #     print("\nBeginning teacher testing...")
        #     results_list.extend(
        #         experiment.test_teachers(
        #             mode=suite.training_params.controls.testing_mode
        #         )
        #     )
        #     print("Teacher testing complete!\n")

        if suite.training_params.controls.run_or_test == "run":
            results_list.extend(experiment.run())
        else:
            results_list.extend(experiment.test_student())

        print(f"Finished experiment {i + 1}/{n}: {experiment.cfg.name.upper()}")
        
        gc.collect()
        torch.cuda.empty_cache()

    suite_scores_dir = suite.directories.get_suite_scores_dir(suite.suite_title)
    suite_scores_dir.mkdir(parents=True, exist_ok=True)

    timestamp = get_timestamp()
    df_path = suite_scores_dir / f"results_df_{timestamp}.parquet"
    results_df = pd.DataFrame(results_list)
    results_df.to_parquet(df_path)
    results_df.to_markdown(suite_scores_dir / f"results_df_{timestamp}.md")

    print(f"Results saved to: {df_path}")
    return results_df


def main(config_path: Optional[str] = None):
    """
    Entry point that handles CLI arguments if no config_path is provided.
    """
    if config_path is None:
        parser = ArgumentParser(description="KDFP Command Line Interface")
        subparsers = parser.add_subparsers(dest="command", help="Available commands")

        run_parser = subparsers.add_parser("run", help="Run the experiment pipeline")
        run_parser.add_argument(
            "-c",
            "--config",
            default=DEFAULT_CONFIG,
            help=f"Path to config file (default: {DEFAULT_CONFIG})",
        )

        args = parser.parse_args()

        if args.command == "run":
            run_suite(args.config)
        else:
            parser.print_help()
    else:
        # If config_path is passed directly (module usage)
        run_suite(config_path)


if __name__ == "__main__":
    main()
