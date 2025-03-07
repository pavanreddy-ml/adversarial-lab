from abc import ABC, abstractmethod
from adversarial_lab.core.types import TensorType

class Preprocessing(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def preprocess(self, sample: TensorType) -> TensorType:
        pass
