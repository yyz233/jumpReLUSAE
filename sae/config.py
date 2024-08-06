from dataclasses import dataclass

from simple_parsing import Serializable, list_field

@dataclass
class TrainerConfig(Serializable):
    """
    Configuration for training a sparse autoencoder on a language model.
    """

    lr: float = 5e-5
    """learning rate"""

    lr_warmup_steps: int = 1000

    sparsity_coefficient: float = 2.0

    epoch: int = 10
    """train epoch"""

    batch_size: int = 1
    """batch size"""

    save_every: int = 1000
    """every x steps to save the sae"""
    
    name: str | None=None
    """train name(to save)"""

    hook_name: str | None=None
    """name of the hook point"""

    save_dir: str | None=None
    """dir to save checkpoint"""

    wandb_project: str | None=None
    """wandb project name"""


