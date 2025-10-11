from argparse import ArgumentParser
from kdfp.schemas.globalConfig import GlobalConfig
import pandas as pd
import os
import torch
# from core.initalization import init_teacher, init_offline_projector
# from core.analysis import analyze

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
    
    # grab config and set root save dir
    config = GlobalConfig.from_yaml(config_path)
    # save_dir = "./results"

    columns = ["experiment"]
    columns.extend(config.tasks.keys())
    result_df = pd.DataFrame(columns=columns)

    experiment_list, experiment_names = config.generate_experiment_list()
    print(f"A total of {len(experiment_names)} experiments have been specified:")
    for i, name in enumerate(experiment_names):
        assert experiment_list[i].name == name, f"Experiment name mismatch: {experiment_list[i].name} != {name}"
        print(f"    {i+1}: {name}")


    for i, experiment in enumerate(experiment_list):
        print(f"Starting experiment {i}: {experiment.name.upper()}")
        experiment.init()
        break
    #     results = experiment.run()

    #     new_row = pd.DataFrame([dict(zip(result_df.columns.tolist(), results))])
    #     result_df = pd.concat([result_df, new_row], ignore_index=True)

    #     print(f"Finished experiment {i}: {experiment.name.upper()}")

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