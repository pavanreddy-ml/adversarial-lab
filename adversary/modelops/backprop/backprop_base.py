from abc import ABC, abstractmethod
from typing import Literal
import torch
import tensorflow as tf

class BackpropagationBase(ABC):
    def __init__(self, 
                 model, 
                 loss):
        pass
    
    @abstractmethod
    def run(self):
        pass
