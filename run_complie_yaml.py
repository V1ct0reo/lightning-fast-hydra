import os

import dotenv
import hydra
from omegaconf import DictConfig
# load environment variables from `.env` file if it exists
# recursively searches for `.env` in all folders starting from work dir
dotenv.load_dotenv(override=True)

@hydra.main(config_path="configs/", config_name="rethink.yaml")
def main(config: DictConfig):

    # Imports should be nested inside @hydra.main to optimize tab completion
    # Read more here: https://github.com/facebookresearch/hydra/issues/934
    from src.train import train
    from src.utils import utils
    log = utils.get_logger(__name__)

    # A couple of optional utilities:
    # - disabling python warnings
    # - easier access to debug mode
    # - forcing debug friendly configuration
    # You can safely get rid of this line if you don't want those
    #utils.extras(config)

    # Pretty print config using Rich library
   # if config.get("print_config"):
    #    utils.print_config(config, resolve=True)
    from hydra.core.hydra_config import HydraConfig
    log.info(f'HydraConfig.get().job:\n{HydraConfig.get().job}\n')
    log.info(f'hydra.utils.get_original_cwd()\n{hydra.utils.get_original_cwd()}\n')
    print(f"to_absolute_path('foo')   : {hydra.utils.to_absolute_path('foo')}")
    print(f"to_absolute_path('/foo')  : {hydra.utils.to_absolute_path('/foo')}")
    print(f"config.name  : {config.run_name}")
    print(f"os.cwd  : {os.getcwd()}")
    for k,v in dict(config).items():
        print(k,':\t',v)
    #print(os.environ['CUDA_VISIBLE_DEVICES'])



if __name__ == "__main__":
    main()
