import os
import random
import logging
from pathlib import Path
from glob import glob
from typing import Optional, Sequence, Literal

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf, ListConfig
import rich.syntax
import rich.tree


def seed_everything(seed: int = 1996):
    """
    Set tunable seed for all the required frameworks
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_logger(filename):
    logger = logging.getLogger("logger")
    logger.setLevel(logging.INFO)
    logging.basicConfig(format="%(message)s", level=logging.DEBUG)
    handler = logging.FileHandler(filename)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter("%(asctime)s:%(levelname)s: %(message)s"))
    logging.getLogger().addHandler(handler)
    return logger


logger = get_logger(__name__)


def extras(config: DictConfig) -> None:
    """Apply optional utilities, controlled by config flags.

    Utilities:
    - Ignoring python warnings
    - Rich config printing

    Args:
        config: the configuration dictionary
    """
    # disable python warnings if <config.ignore_warnings=True>
    if config.get("ignore_warnings"):
        logger.info("Disabling python warnings! <config.ignore_warnings=True>")

    # pretty print config tree using Rich library if <config.print_config=True>
    if config.get("print_config"):
        logger.info("Printing config tree with Rich! <config.print_config=False>")
        print_config(config, resolve=False)


def print_config(
    config: DictConfig,
    print_order: Sequence[str] = (
        "dataset",
        "model",
        "trainer",
        "callbacks",
        "logger",
    ),
    resolve: Optional[bool] = True,
) -> None:
    """
    Print content of DictConfig using Rich library and its tree structure.

    Args:
        config (DictConfig): Configuration composed by Hydra.
        print_order (Sequence[str], optional): Determines in what order config components are printed.
        resolve (bool, optional): Whether to resolve reference fields of DictConfig.
    """
    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    quee = []

    for field in print_order:
        quee.append(field) if field in config else logger.info(f'Field "{field}" not found in config')

    for field in config:
        if field not in quee:
            quee.append(field)

    for field in quee:
        branch = tree.add(field, style=style, guide_style=style)

        config_group = config[field]
        if isinstance(config_group, DictConfig):
            branch_content = OmegaConf.to_yaml(config_group, resolve=resolve)
        else:
            branch_content = str(config_group)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    rich.print(tree)

    with open("config_tree.log", "w") as file:
        rich.print(tree, file=file)


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


def get_config_from_path(paths_dict):
    return_dict = {}
    for k, v in paths_dict.items():
        # print("==>", type(v), v)
        if type(v) == ListConfig:
            for vid, vi in enumerate(v):
                if vid == 0:
                    return_dict[k] = glob(vi, recursive=True)
                else:
                    return_dict[k] += glob(vi, recursive=True)
        else:
            return_dict[k] = glob(v, recursive=True)
    return return_dict


def make_project_dicts(output_root, project_name):
    root = Path(output_root)
    project_path = Path(os.path.join(output_root, project_name)).mkdir(parents=True, exist_ok=True)
    return project_path
