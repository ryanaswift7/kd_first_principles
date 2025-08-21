from argparse import ArgumentParser
from schemas.globalConfig import GlobalConfig
def parse_args():
    parser = ArgumentParser("Run experiment pipeline.")
    parser.add_argument(
        "--config", "-c",
        required=True,
        help="Path to YAML configuration file",
    )
    return parser.parse_args()


def main(config_path: str):

    config = GlobalConfig.from_yaml(config_path)

    experiment_list = config.generate_experiment_list()


if __name__ == "__main__":
    args = parse_args
    main(config_path=args.config)
