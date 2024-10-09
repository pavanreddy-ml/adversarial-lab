


class IncompatibilityError(Exception):
    def __init__(self, 
                 message: str = "An incompatibility error occurred", 
                 details: str = None):
        self.message = message
        self.details = details
        super().__init__(self.message)

    def __str__(self):
        if self.details:
            return f"{self.message}\nDetails: {self.details}"
        return self.message
    

class VectorDimensionsError(Exception):
    def __init__(self, 
                 message: str = "An Vector Dimensions error occurred", 
                 details: str = None):
        self.message = message
        self.details = details
        super().__init__(self.message)

    def __str__(self):
        if self.details:
            return f"{self.message}\nDetails: {self.details}"
        return self.message
