from abc import ABC, abstractmethod


class DB(ABC):
    def __init__(self, *args, **kwargs) -> None:
        pass
    
    @abstractmethod
    def validate_connection(self) -> bool:
        pass

    @abstractmethod
    def create_table(self) -> None:
        pass

    @abstractmethod
    def delete_table(self) -> None:
        pass
    
    @abstractmethod
    def insert(self, data) -> None:
        pass