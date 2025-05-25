# minigrad
Minigrad is a lightweight autograd engine for educational purposes.

## Installation
```bash
pip install .
```

## Features
- NumPy-based tensors
- Forward and backward pass
- Basic neural net layers

## Quick Start

```python
from minigrad import Tensor

x = Tensor([1.0, 2.0], requires_grad=True)
y = (x * 2).sum()
y.backward()
print(x.grad)
```

## Attention Helper

The library includes a convenience function for scaled dot-product attention:

```python
from minigrad import Tensor, scaled_dot_product_attention
import numpy as np

q = Tensor(np.random.randn(2, 4, 8), requires_grad=True)
out = scaled_dot_product_attention(q, q, q, causal=True)
```
Use `causal=True` to apply an upper triangular mask for autoregressive models.

## Directory Structure
```text
minigrad/
├── tensor.py         # Core Tensor class
├── functions.py      # Core operations (Function subclasses)
├── nn/
│   ├── module.py     # AIPModule, AIPParameter
│   ├── layers.py     # Linear, Embedding, LayerNorm
│   └── positional.py # Positional encodings
examples/
tests/
scripts/
```
## Testing
To run the unit tests:
```bash
pytest tests/
```

## Acknowledgements
- [micrograd](https://github.com/karpathy/micrograd)
- [Pytorch](https://pytorch.org/)
- [NumPy](https://numpy.org/)
