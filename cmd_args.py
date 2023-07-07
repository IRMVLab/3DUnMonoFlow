import yaml

from utils.easydict import EasyDict

def parse_args_from_yaml(yaml_path):
    with open(yaml_path, 'r') as fd:
        args = yaml.safe_load(fd)
        args = EasyDict(d=args)

    return args

