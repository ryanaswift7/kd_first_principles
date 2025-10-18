from argparse import ArgumentParser
from pathlib import Path
from kdfp.schemas.globalConfig import GlobalConfig
import pandas as pd
import os
import torch

DEFAULT_CONFIG = 'kdfp/config.yaml'

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

    print("\nBeginning baseline testing...")
    results_list = config.run_baseline_tests()
    print("Baseline testing complete!")

    print("\nStarting experiment runs...")
    for i, experiment in enumerate(experiment_list):
        print(f"Starting experiment {i+1}/{n}: {experiment.name.upper()}")
        experiment.init()
        results_list.extend(experiment.run())
        print(f"Finished experiment {i+1}/{n}: {experiment.name.upper()}")
        # break

    base_scores_dir = Path(config.directories.root + '/' + config.directories.scores).resolve()
    base_scores_dir.mkdir(parents=True, exist_ok=True)
    df_path = base_scores_dir / "results_df.parquet"
    results_df = pd.DataFrame(results_list)
    results_df.to_parquet(df_path)
    print(f"Results dataframe saved in Parquet format to:\n\t{df_path}")
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