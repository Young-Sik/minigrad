import numpy as np
import minigrad.tensor as mini
from .module import AIPModule, AIPParameter

class AIPEmbedding(AIPModule):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.embeddings = AIPParameter(np.random.randn(num_embeddings, embedding_dim))

    def __call__(self, indices):
        return self.forward(indices)

    def forward(self, indices):
        return self.embeddings[indices]

class AIPLinear(AIPModule):
    """Fully connected linear layer."""
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias

        std = 1.0 / np.sqrt(in_features)
        self.weight = AIPParameter(np.random.randn(out_features, in_features) * std)
        if self.use_bias:
            self.bias = AIPParameter(np.zeros(out_features))

    def forward(self, x: mini.Tensor) -> mini.Tensor:
        y = x @ self.weight.T
        if self.use_bias:
            y = y + self.bias
        return y

    __call__ = forward

    def extra_repr(self):
        return (f"in_features={self.in_features}, out_features={self.out_features}, "
                f"bias={self.use_bias}")

class AIPLayerNorm(AIPModule):
    """Layer Normalization."""
    def __init__(self, normalized_shape, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.affine = affine

        if self.affine:
            self.weight = AIPParameter(np.ones(self.normalized_shape))
            self.bias = AIPParameter(np.zeros(self.normalized_shape))

    def forward(self, x: mini.Tensor) -> mini.Tensor:
        dims = tuple(range(-len(self.normalized_shape), 0))
        mean = x.mean(axis=dims, keepdims=True)
        var = x.var(axis=dims, keepdims=True)
        inv_std = (var + self.eps) ** -0.5
        x_norm = (x - mean) * inv_std
        if self.affine:
            x_norm = x_norm * self.weight + self.bias
        return x_norm

    __call__ = forward

    def extra_repr(self):
        return (f"normalized_shape={self.normalized_shape}, eps={self.eps}, "
                f"affine={self.affine}")
