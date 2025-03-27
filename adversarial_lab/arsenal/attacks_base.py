from abc import ABC, abstractmethod


class AttacksBase(ABC):
    @abstractmethod
    def attack(self, *args, **kwargs):
        pass