from abc import ABC, abstractmethod
from typing import Literal
import torch
import tensorflow as tf

class ModelInfoBase(ABC):
    def __init__(self, 
                 model, 
                 loss):
        pass
    
    @abstractmethod
    def get_info(self):
        pass
