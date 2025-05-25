import numpy as np
import minigrad.functions as F

class Tensor:
    def __init__(self, data, requires_grad=False, _ctx=None):
        self.data = np.asarray(data, dtype=float)
        self.requires_grad = requires_grad
        self.grad = None            # accumulated gradient
        self._ctx = _ctx            # Function that created this tensor

    # -------- iterative backward (topo queue) --------
    def backward(self, grad=None):
        if not self.requires_grad:
            return
        if grad is None:
            if self.data.size != 1:
                raise RuntimeError("grad must be specified for non-scalar tensor")
            grad = np.ones_like(self.data)

        # 1) build reverse‑topological order (post‑order DFS)
        topo, visited = [], set()
        def build(t):
            if t not in visited:
                visited.add(t)
                if t._ctx:
                    for p in t._ctx.parents:
                        build(p)
                topo.append(t)
        build(self)

        # 2) seed gradient map
        self.grad = grad

        # 3) iterate once in reverse topo order
        for t in reversed(topo):
            g_out = t.grad
            
            # there's no need to backprop
            if g_out is None or t._ctx is None:
                continue
            
            # propagate to parents
            for parent, g in zip(t._ctx.parents, t._ctx.backward(g_out)):
                if g is None:
                    continue
                parent.grad = g if parent.grad is None else parent.grad + g
            
            # free up memory
            # t._ctx.saved_tensors = ()   # release cached forward intermediates
            # t._ctx = None               # break graph reference so GC can collect

    def zero_grad(self):
        self.grad = None

    def __repr__(self):
        if self.data.ndim == 0:
            return f"Tensor({self.data})"
        arr = np.array2string(self.data, formatter={'float_kind': lambda x: f'{x:.4f}'})
        return f"Tensor: {arr}"

    # -------- operator overloads --------
    def __add__(self, o): return F.Add.apply(self, o if isinstance(o, Tensor) else Tensor(o))
    def __mul__(self, o): return F.Mul.apply(self, o if isinstance(o, Tensor) else Tensor(o))
    def __matmul__(self, o): return F.MatMul.apply(self, o if isinstance(o, Tensor) else Tensor(o))
    def __sub__(self, o):    return self + (-o)
    def __neg__(self):       return self * -1
    def __truediv__(self, o):
        o = o if isinstance(o, Tensor) else Tensor(o)
        return self * (o ** -1)
    def __pow__(self, p):   return F.Pow.apply(self, Tensor(p))
    def __getitem__(self, idx): return F.Gather.apply(self, idx)
    
    def __radd__(self, o): return self + o
    def __rmul__(self, o): return self * o
    def __rsub__(self, o): return (-self) + o
    def __rtruediv__(self, o): return Tensor(o) * (self ** -1)
    def __rmatmul__(self, o): return Tensor(o) @ self

    # -------- convenience helpers --------
    def exp(self):            return F.Exp.apply(self)
    def sum(self, axis=None, keepdims=False):    return F.Sum.apply(self, axis=axis, keepdims=keepdims)
    def mean(self, axis=None, keepdims=False):   return F.Mean.apply(self, axis=axis, keepdims=keepdims)
    def var(self, axis=None, keepdims=False):    return F.Var.apply(self, axis=axis, keepdims=keepdims)

    def relu(self):            return F.ReLU.apply(self)
    def norm(self):            return F.Norm.apply(self)
    def softmax(self, axis=-1): return F.Softmax.apply(self, axis)
    def logsumexp(self, axis=-1, keepdims=False): return F.LogSumExp.apply(self, axis, keepdims)
    def cross_entropy(self, t): return F.CrossEntropy.apply(self, t)

    # properties
    @property
    def T(self): return F.Transpose.apply(self)

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        return F.Reshape.apply(self, shape)

    def permute(self, *axes):
        if len(axes) == 1 and isinstance(axes[0], (tuple, list)):
            axes = axes[0]
        return F.Permute.apply(self, axes)
    
