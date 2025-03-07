from . import Adam, Optimizer

class OptimizerRegistry:
    __optmizers = {
        "adam": Adam
    }

    @classmethod
    def get(cls, optmizer: str) -> Optimizer:
        if optmizer not in cls.__optmizers:
            raise ValueError(f"Invalid value for optmizer: '{optmizer}'. Optimizer of this type does not exist.")
        return cls.__optmizers[optmizer]
