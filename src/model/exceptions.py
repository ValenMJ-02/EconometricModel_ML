"""
Module containing custom exception classes for the project.
"""

class InvalidSplitSizeError(ValueError):
    """
    Exception raised when the test_size parameter for data splitting is not between 0 and 1.

    Attributes:
        test_size (float): The invalid test size provided.
        message (str): Explanation of the error.
    """
    def __init__(self, test_size: float) -> None:
        message = f"Test size must be between 0 and 1, got {test_size}."
        super().__init__(message)
        self.test_size: float = test_size
