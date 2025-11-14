from argparse import ArgumentParser
import datetime
from pathlib import Path
from kdfp.schemas.globalConfig import GlobalConfig
import pandas as pd
import os
import torch

from kdfp.utils.general import get_timestamp

DEFAULT_CONFIG = 'kdfp/configs/base.yaml'

def parse_cli_args():
    """Parse CLI arguments for the kdfp command"""
    parser = ArgumentParser(description='KDFP Command Line Interface')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Add 'run' command
    run_parser = subparsers.add_parser('run', help='Run the experiment pipeline')
    run_parser.add_argument('-c', '--config', default=DEFAULT_CONFIG, 
                            help=f"Path to config file (default: {DEFAULT_CONFIG})")
    
    return parser.parse_args()


def main(config_path: str):
    os.environ["TOKENIZERS_PARALLELISM"] = 'true'
    torch.set_float32_matmul_precision('medium')
    
    config = GlobalConfig.from_yaml(config_path)

    experiment_list, experiment_names = config.generate_experiment_list()
    n = len(experiment_names)
    print(f"A total of {n} experiments have been specified:")
    for i, name in enumerate(experiment_names):
        assert experiment_list[i].name == name, f"Experiment name mismatch: {experiment_list[i].name} != {name}"
        print(f"    {i+1}: {name}")

    results_list = []
    
    if config.training_params.get_baselines:
        print("\nBeginning baseline testing...")
        results_list.extend(config.run_baseline_tests())
        print("Baseline testing complete!")

    print("\nStarting experiment runs...")
    for i, experiment in enumerate(experiment_list):
        if "narrow" not in experiment.name:
            print(f"Skipping experiment {i+1}/{n}: {experiment.name.upper()}")
            continue

        print(f"\nStarting experiment {i+1}/{n}: {experiment.name.upper()}")
        experiment.init()

        # Now that teachers have certainly been initialized, get benchmark scores.
        # We can get them from any experiment and only need to do it once, might as well
        # do the first one

        if i == 0 and config.training_params.get_baselines:
            print(f"\nBeginning teacher testing...")
            results_list.extend(experiment.test_teachers(mode=config.training_params.testing_mode))
            print(f"Teacher testing complete!\n")

        if config.training_params.run_or_test == 'run':
            # use this to create, then test models
            results_list.extend(experiment.run())  
        else:
            # use this to test existing models
            results_list.extend(experiment.test_student(mode=config.training_params.testing_mode))  

        print(f"Finished experiment {i+1}/{n}: {experiment.name.upper()}")

    base_scores_dir = Path(config.directories.root + '/' + config.directories.scores).resolve()
    base_scores_dir.mkdir(parents=True, exist_ok=True)
    df_path = base_scores_dir / f"results_df_{get_timestamp()}.parquet"
    results_df = pd.DataFrame(results_list)
    results_df.to_parquet(df_path)
    results_df.to_markdown(base_scores_dir / f"results_df_{get_timestamp()}.md")
    print(f"Results dataframe for experiment configuration {config_path} saved in Parquet format to:\n\t{df_path}\n")
    print(results_df)

    #     new_row = pd.DataFrame([dict(zip(result_df.columns.tolist(), results))])
    #     result_df = pd.concat([result_df, new_row], ignore_index=True)


    # columns.extend(["average", "relative lift over baseline"])
    # analyze(result_df)

def cli_main():
    """Entry point for the kdfp command"""
    args = parse_cli_args()
    
    if args.command == 'run':
        main(config_path=args.config)
    else:
        print("Usage: kdfp run [--config CONFIG_FILE]")
        print("Run 'kdfp run --help' for more options")

if __name__ == "__main__":
    cli_main()