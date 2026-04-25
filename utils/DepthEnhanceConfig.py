from engine.Config import Config
from omegaconf import OmegaConf
from typing import List, Dict, Any, Sequence

from utils.common import parse_config_file_arg


class DepthEnhanceConfig(Config):
    def __init__(self, config_file: str | List[str] | None = None, cli_overrides: Dict[str, Any] | None = None):
        if config_file is not None:
            if isinstance(config_file, str):
                paths: Sequence[str] = parse_config_file_arg(config_file)
            else:
                paths = config_file
            yaml_cfg = OmegaConf.create()
            for p in paths:
                yaml_cfg = OmegaConf.merge(yaml_cfg, OmegaConf.load(p))
        else:
            yaml_cfg = OmegaConf.create()
        if cli_overrides is not None:
            cli_cfg = OmegaConf.from_cli(cli_overrides)
        else:
            cli_cfg = OmegaConf.create()
        self.config = OmegaConf.merge(yaml_cfg, cli_cfg)
        OmegaConf.set_readonly(self.config, True)