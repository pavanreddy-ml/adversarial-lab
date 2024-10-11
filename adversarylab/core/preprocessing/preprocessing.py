from abc import ABC, ABCMeta, abstractmethod
from typing import Literal, Union, TypeVar, Generic

class Preprocessing(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def preprocess(self, input):
        pass
