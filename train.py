import dotenv
import hydra
from omegaconf import DictConfig, OmegaConf
from typing import Optional

# load environment variables from `.env` file if it exists
# recursively searches for `.env` in all folders starting from work dir
dotenv.load_dotenv(override=True)


@hydra.main(version_base=None, config_path="config/", config_name="train.yaml")
def train(config: DictConfig) -> Optional[float]:
    """
    Hydra based training pipeline inspired by Ashleev's hydra template.
    """

    from mdx import utils
    from mdx.training import train

    # Optional utilities gets set using this module
    utils.extras(config)

    # Calling the train function from src/mdx
    return train(config)


if __name__ == "__main__":
    train()
