import draccus
from dataclasses import dataclass
from pathlib import Path
from typing import Union

@dataclass
class Cfg:
    model_family: str = "openvla"
    pretrained_checkpoint: Union[str, Path] = ""
    task_suite_name: str = "libero_spatial"
    center_crop: bool = True
    num_trials_per_task: int = 10
    seed: int = 7
    use_wandb: bool = False

@draccus.wrap()
def main(cfg: Cfg):
    print("CHECKPOINT_ARG =", repr(cfg.pretrained_checkpoint))

main()
