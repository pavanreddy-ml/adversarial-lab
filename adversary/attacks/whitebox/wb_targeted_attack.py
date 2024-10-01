from typing import Dict, Union
from torch.nn import Module as TorchModel
from adversary.core.losses.loss_base import Loss
from adversary.core.optimizers.optimizer_base import Optimizer
from tensorflow.keras.models import Model as TFModel
import tensorflow as tf
from . import WhiteBoxAttack


class TargetedWhiteBoxAttack(WhiteBoxAttack):
    def __init__(self, 
                 model: Union[TorchModel, TFModel], 
                 loss: str | Loss, 
                 optimizer: str | Optimizer, 
                 optimizer_params: Dict | None = None, 
                 *args, 
                 **kwargs) -> None:
        super().__init__(model, loss, optimizer, optimizer_params, *args, **kwargs)

    def attack(self,
               sample,
               target_class,
               target_vector=None,
               *args,
               **kwargs):
        pass