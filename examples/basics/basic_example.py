import numpy as np
from minigrad import Tensor

# Create a tensor with gradient tracking
x = Tensor([1.0, 2.0, 3.0], requires_grad=True)

# Perform a simple computation
y = (2 * x + 1).sum()

# Backpropagate to compute gradients
y.backward()

print("Input:", x)
print("Gradient:", x.grad)

