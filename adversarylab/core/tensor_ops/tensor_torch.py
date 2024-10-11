import torch
from typing import Optional, Dict, Any, Union

class TensorOpsTorch():
    def __init__(self, *args, **kwargs) -> None:
        pass

    def tensor(self, 
               shape: Union[tuple, list], 
               distribution: str = "fill", 
               dist_params: Optional[Dict[str, Union[int, float]]] = None) -> torch.Tensor:
        raise NotImplementedError("'tensor' not Implemented for Torch")

    def tensor_like(self, 
                    array: Union[torch.Tensor, list, tuple], 
                    distribution: str = "fill", 
                    dist_params: Optional[Dict[str, Union[int, float]]] = None) -> torch.Tensor:
        raise NotImplementedError("'tensor_like' not Implemented for Torch")

    def to_tensor(self, 
                  data: Any, 
                  dtype: torch.dtype = torch.float32) -> torch.Tensor:
        raise NotImplementedError("'to_tensor' not Implemented for Torch")