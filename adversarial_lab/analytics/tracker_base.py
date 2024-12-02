from typing import Dict
from abc import ABC, abstractmethod


class Tracker(ABC):
    def __init__(self,
                 track_batch: bool = True,
                 track_epoch: bool = True,) -> None:
        
        if not hasattr(self, 'columns') or not isinstance(self.columns, dict):
            raise AttributeError("The 'columns' attribute must be defined as a dictionary in the subclass.")

        self.track_batch = track_batch
        self.track_epoch = track_epoch

    def pre_training(self,
                     *args,
                     **kwargs
                     ) -> None:
        pass

    def post_batch(self,
                   batch_num: int,
                   *args,
                   **kwargs
                   ) -> None:
        pass

    def post_epoch(self,
                   epoch_num: int,
                   *args,
                   **kwargs
                   ) -> None:
        pass

    def post_training(self,
                      *args,
                      **kwargs
                      ) -> None:
        pass

    def get_columns(self) -> Dict:
        return self.columns
    
    @abstractmethod
    def serialize(self) -> Dict:
        pass

    @abstractmethod
    def reset_values(self) -> None:
        pass
