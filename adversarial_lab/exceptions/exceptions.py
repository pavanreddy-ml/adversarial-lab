from typing import Optional


class IncompatibilityError(Exception):
    def __init__(self, 
                 message: str = "", 
                 details: str = None):
        self.message = f"""
        An Incompatibility error occurred: {message}.

        This error typically occurs when the attack is not compatible with a particular 'Tracker', 'Loss', 
        'NoiseGenerator' or any other component.
        """
        self.details = details
        super().__init__(self.message)

    def __str__(self):
        if self.details:
            return f"{self.message}\nDetails: {self.details}"
        return self.message
    

class VectorDimensionsError(Exception):
    def __init__(self, 
                 message: str = "", 
                 details: str = None):
        self.message = f"""
        A Vector Dimensions error occurred: {message}.

        This error typically occurs when the input vector dimensions do not match the expected 
        dimensions of the model or layer or underlying operation..
        """
        self.details = details
        super().__init__(self.message)

    def __str__(self):
        if self.details:
            return f"{self.message}\nDetails: {self.details}"
        return self.message


class IndifferentiabilityError(Exception):
    def __init__(self, 
                 message: str = "", 
                 details: str = None):
        self.message = f"""
        An Indifferentiability error occurred: {message}.

        One or more components in the attack pipeline are not differentiable.
        Follow the following steps to debug and identify the component:

        1. If 'on_original' is set to True, ensure that the preprocessing layer is fully differentiable. To debug this, 
        set 'on_original' to False and check if the error persists. If the error disappears, the preprocessing layer is 
        likely the issue.

        2. If you are using a custom Loss function, ensure that it is differentiable. All loss functions provided by the 
        library are differentiable. This may be an issue only if you are using a custom loss function. To debug this, use
        a loss function provided by the library like 'CategoricalCrossEntropy' or 'BinaryCrossEntropy' and check if the 
        error persists. If the error disappears, the custom loss function is likely the issue.

        """ 
        self.details = details
        super().__init__(self.message)

    def __str__(self):
        if self.details:
            return f"{self.message}\nDetails: {self.details}"
        return self.message