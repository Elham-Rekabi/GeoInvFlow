"""Simple forward functions for toy inversion examples."""


def quadratic_function(x1, x2):
    
    """Evaluate f(x1, x2) = (x1 - 1)^2 + (x2 - 2)^2.
    Accepts scalars, NumPy arrays, or PyTorch tensors .
    """
    return (x1 - 1) ** 2 + (x2 - 2) ** 2
