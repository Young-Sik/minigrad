import numpy as np
import minigrad.tensor as mini

#############################
# Function base classes     #
#############################

class Function:
    def __init__(self, *parents):
        self.parents = parents
        self.saved_tensors = ()
    def save_for_backward(self, *tensors):
        self.saved_tensors = tensors
    def backward(self, grad_out):
        raise NotImplementedError
    
    # -------- utility shared by all ops --------
    @staticmethod
    def any_requires_grad(*inputs):
        """Return True if any input Tensor participates in grad."""
        return any(getattr(t, 'requires_grad', False) for t in inputs)
    
    def __repr__(self):
        return f"{self.__class__.__name__}({', '.join(map(str, self.parents))})"
    
    ## -------- utility for broadcasting gradients --------
    def save_shapes(self, *inputs):
        self.shapes = [i.data.shape for i in inputs]
    def apply_unbroadcast(self, grads):
        return [unbroadcast(g, s) if g is not None else None for g, s in zip(grads, self.shapes)]


#############################
# Helper                    #
#############################
def unbroadcast(grad, target_shape):
    """Reverse NumPy broadcasting for gradients.
    Handles scalar (0-D) grads safely.
    """
    grad = np.asarray(grad)
    # If both are scalars, nothing to do
    if grad.ndim == 0 and len(target_shape) == 0:
        return grad.astype(float)
    # Reduce leading broadcast axes
    while grad.ndim > len(target_shape):
        grad = grad.sum(axis=0)
    # Collapse singleton dims that were broadcast
    for axis, dim in enumerate(target_shape):
        if dim == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad.reshape(target_shape)

#############################
# Basic element‑wise ops    #
#############################
class Add(Function):
    @staticmethod
    def apply(a, b):
        out = mini.Tensor(a.data + b.data, requires_grad=Add.any_requires_grad(a, b))
        ctx = Add(a, b); ctx.save_shapes(a, b); out._ctx = ctx
        return out
    def backward(self, g):
        return self.apply_unbroadcast([g, g])

class Mul(Function):
    @staticmethod
    def apply(a, b):
        out = mini.Tensor(a.data * b.data, requires_grad=Mul.any_requires_grad(a, b))
        ctx = Mul(a, b); ctx.save_shapes(a, b); ctx.save_for_backward(a, b); out._ctx = ctx
        return out
    def backward(self, g):
        a, b = self.saved_tensors
        grads = [g * b.data if a.requires_grad else None,
                 g * a.data if b.requires_grad else None]
        return self.apply_unbroadcast(grads)

#############################
# MatMul / Pow             #
#############################

class MatMul(Function):
    @staticmethod
    def apply(a, b):
        out = mini.Tensor(np.matmul(a.data, b.data), requires_grad=a.requires_grad or b.requires_grad)
        ctx = MatMul(a, b)
        ctx.save_for_backward(a, b); ctx.save_shapes(a, b); out._ctx = ctx
        return out
    def backward(self, g):
        a, b = self.saved_tensors
        def _T(x):
            if x.ndim < 2:
                return x
            axes = list(range(x.ndim)); axes[-2], axes[-1] = axes[-1], axes[-2]
            return x.transpose(axes)
        if a.data.ndim == b.data.ndim == 1:
            grads = [g * b.data if a.requires_grad else None,
                     g * a.data if b.requires_grad else None]
        elif a.data.ndim == 2 and b.data.ndim == 1:
            # g has shape (m,) where m = a.shape[0]
            grads = [g[:, None] * b.data if a.requires_grad else None,  # outer product ⟹ (m, n)
                      a.data.T @ g if b.requires_grad else None]   # (n,)
        else:
            grads = [np.matmul(g, _T(b.data)) if a.requires_grad else None,
                     np.matmul(_T(a.data), g) if b.requires_grad else None]
        return self.apply_unbroadcast(grads)

class Pow(Function):
    @staticmethod
    def apply(a, b):
        out = mini.Tensor(a.data ** b.data, requires_grad=a.requires_grad)
        ctx = Pow(a, b); ctx.save_for_backward(a, b); out._ctx = ctx
        return out
    def backward(self, g):
        a, b = self.saved_tensors
        grad_a = g * b.data * (a.data ** (b.data - 1)) if a.requires_grad else None
        return [grad_a, None]

class Exp(Function):
    @staticmethod
    def apply(a):
        out = mini.Tensor(np.exp(a.data), requires_grad=a.requires_grad)
        ctx = Exp(a); ctx.save_for_backward(a); out._ctx = ctx
        return out
    def backward(self, g):
        (a,) = self.saved_tensors
        return [g * np.exp(a.data)]

def exp(x):
    """mini.exp(x) → Tensor"""
    return Exp.apply(x)

#############################
# Reductions & Activations  #
#############################

class Sum(Function):
    @staticmethod
    def apply(a, axis=None, keepdims=False):
        out = mini.Tensor(a.data.sum(axis=axis, keepdims=keepdims), requires_grad=a.requires_grad)
        ctx = Sum(a)
        ctx.axis = axis; ctx.keepdims = keepdims; ctx.input_shape = a.data.shape; out._ctx = ctx
        return out
    def backward(self, g):
        expanded = g
        if not self.keepdims and self.axis is not None:
            expanded = np.expand_dims(g, axis=self.axis)
        return [np.broadcast_to(expanded, self.input_shape)]

class Mean(Function):
    @staticmethod
    def apply(a, axis=None, keepdims: bool = False):
        out_data = a.data.mean(axis=axis, keepdims=keepdims)
        out = mini.Tensor(out_data, requires_grad=a.requires_grad)

        ctx = Mean(a)
        ctx.axis = axis
        ctx.keepdims = keepdims
        ctx.input_shape = a.data.shape
        # number of elements reduced → scaling factor for backward
        if axis is None:
            ctx.count = a.data.size
        else:
            ax_tuple = axis if isinstance(axis, (tuple, list)) else (axis,)
            ctx.count = np.prod([a.data.shape[ax] for ax in ax_tuple])
        out._ctx = ctx
        return out

    def backward(self, g_out):
        # broadcast gradient back to input shape, then scale by 1/N
        if not self.keepdims and self.axis is not None:
            g_out = np.expand_dims(g_out, axis=self.axis)
        grad = np.broadcast_to(g_out, self.input_shape) / self.count
        return [grad]

# ----------------------------------------------------------------------
# Variance : population variance (LayerNorm friendly) -------------------
# ----------------------------------------------------------------------
class Var(Function):
    @staticmethod
    def apply(a, axis=None, keepdims: bool = False):
        mean_keep = a.data.mean(axis=axis, keepdims=True)
        var_data = ((a.data - mean_keep) ** 2).mean(axis=axis, keepdims=keepdims)
        out = mini.Tensor(var_data, requires_grad=a.requires_grad)

        ctx = Var(a)
        ctx.axis = axis
        ctx.keepdims = keepdims
        ctx.input_shape = a.data.shape
        ctx.mean_keep = mean_keep
        ax_tuple = axis if (axis is not None and not isinstance(axis, tuple)) else axis
        if ax_tuple is None:
            ctx.count = a.data.size
        else:
            ax_tuple = ax_tuple if isinstance(ax_tuple, tuple) else (ax_tuple,)
            ctx.count = np.prod([a.data.shape[ax] for ax in ax_tuple])
        return out

    def backward(self, g_out):
        x, = self.parents
        if not self.keepdims and self.axis is not None:
            g_out = np.expand_dims(g_out, axis=self.axis)
        grad = 2.0 * (x.data - self.mean_keep) / self.count * g_out
        return [grad]

class ReLU(Function):
    @staticmethod
    def apply(a):
        out = mini.Tensor(np.maximum(0, a.data), requires_grad=a.requires_grad)
        ctx = ReLU(a); ctx.save_for_backward(a); out._ctx = ctx
        return out
    def backward(self, g):
        (a,) = self.saved_tensors
        return [g * (a.data > 0)]

class Norm(Function):
    @staticmethod
    def apply(a):
        n = np.linalg.norm(a.data)
        out = mini.Tensor(n, requires_grad=a.requires_grad)
        ctx = Norm(a); ctx.save_for_backward(a, n); out._ctx = ctx
        return out
    def backward(self, g):
        a, n = self.saved_tensors
        return [np.zeros_like(a.data) if n == 0 else g * a.data / n]

#############################
# LogSumExp                 #
#############################

class Softmax(Function):
    """Softmax (default: last axis)."""
    @staticmethod
    def apply(t, axis=-1):
        shifted = t.data - t.data.max(axis=axis, keepdims=True)
        e = np.exp(shifted)
        p = e / e.sum(axis=axis, keepdims=True)
        out = mini.Tensor(p, requires_grad=t.requires_grad)
        ctx = Softmax(t); ctx.axis = axis; ctx.save_for_backward(out); out._ctx = ctx
        return out
    def backward(self, g):
        (p,) = self.saved_tensors
        axis = self.axis
        dot = (g * p.data).sum(axis=axis, keepdims=True)
        return [p.data * (g - dot)]

    
class CrossEntropy(Function):
    """Cross-entropy with logits (Tensor) and integer targets (Tensor or ndarray)."""
    @staticmethod
    def apply(logits, target):
        if not isinstance(target, mini.Tensor):
            target = mini.Tensor(target, requires_grad=False)

        shifted_logits = logits.data - logits.data.max(axis=-1, keepdims=True)  # numerically stable shift
        exp_logits = np.exp(shifted_logits)
        log_probs = shifted_logits - np.log(exp_logits.sum(axis=-1, keepdims=True))
        
        N = target.data.shape[0]
        idx = np.arange(N), target.data.astype(int)
        loss = -log_probs[idx].mean()  # Mean negative log-likelihood

        out = mini.Tensor(loss, requires_grad=True)

        ctx = CrossEntropy(logits, target)
        probs = exp_logits / exp_logits.sum(axis=-1, keepdims=True)
        ctx.save_for_backward(probs.copy(), target.data.astype(int))  # copy for safe backward
        ctx.batch_size = N

        out._ctx = ctx
        return out

    def backward(self, grad_out):
        probs, tgt = self.saved_tensors
        grad_logits = probs.copy()  # avoid in-place modification
        grad_logits[np.arange(self.batch_size), tgt] -= 1  # Subtract 1 for true class (gradient of cross-entropy wrt logits)
        grad_logits /= self.batch_size
        return [grad_out * grad_logits, None]


class LogSumExp(Function):
    @staticmethod
    def apply(x, axis=-1, keepdims=False):
        shifted = x.data - x.data.max(axis=axis, keepdims=True)
        exp_shift = np.exp(shifted)
        sum_exp = exp_shift.sum(axis=axis, keepdims=True)
        out_data = np.log(sum_exp)
        if not keepdims:
            out_data = out_data.squeeze(axis=axis)
        out = mini.Tensor(out_data, requires_grad=x.requires_grad)
        ctx = LogSumExp(x)
        ctx.axis = axis
        ctx.keepdims = keepdims
        ctx.save_for_backward(exp_shift, sum_exp)
        out._ctx = ctx
        return out

    def backward(self, grad_out):
        exp_shift, sum_exp = self.saved_tensors
        softmax_grad = exp_shift / sum_exp
        if not self.keepdims:
            grad_out = np.expand_dims(grad_out, axis=self.axis)
        return [grad_out * softmax_grad]

# --- in mini_autograd_engine.py --------------------------------------------
class Transpose(Function):
    @staticmethod
    def apply(a):
        out = mini.Tensor(a.data.T, requires_grad=a.requires_grad)
        ctx = Transpose(a)
        out._ctx = ctx
        return out
    def backward(self, g):          # g has same shape as out
        (a,) = self.parents
        return [g.T]                # d(a.T)/da  == transpose again

class Reshape(Function):
    @staticmethod
    def apply(a, shape):
        out = mini.Tensor(a.data.reshape(shape), requires_grad=a.requires_grad)
        ctx = Reshape(a)
        ctx.orig_shape = a.data.shape
        out._ctx = ctx
        return out
    def backward(self, g_out):
        return [g_out.reshape(self.orig_shape), None]

class Permute(Function):
    @staticmethod
    def apply(a, axes):
        out = mini.Tensor(np.transpose(a.data, axes), requires_grad=a.requires_grad)
        ctx = Permute(a)
        ctx.axes = axes
        ctx.inv_axes = np.argsort(axes)
        out._ctx = ctx
        return out
    def backward(self, g_out):
        return [np.transpose(g_out, self.inv_axes), None]

def reshape(t, shape):
    """mini.reshape(t, shape) -> Tensor"""
    return Reshape.apply(t, shape)

def permute(t, axes):
    """mini.permute(t, axes) -> Tensor"""
    return Permute.apply(t, axes)

# ----------------------------------------------------------------------
# Concat : np.concatenate wrapper ---------------------------------------
# ----------------------------------------------------------------------
class Concat(Function):
    @staticmethod
    def apply(*tensors, axis: int = 0):
        data_list = [t.data for t in tensors]
        out_data = np.concatenate(data_list, axis=axis)
        requires_grad = any(t.requires_grad for t in tensors)
        out = mini.Tensor(out_data, requires_grad=requires_grad)

        ctx = Concat(*tensors)
        ctx.axis = axis
        ctx.sizes = [d.shape[axis] for d in data_list]
        out._ctx = ctx
        return out

    def backward(self, g_out):
        idx, axis = 0, self.axis
        base_slice = [slice(None)] * g_out.ndim
        grads = []
        for size in self.sizes:
            sl = base_slice.copy()
            sl[axis] = slice(idx, idx + size)
            grads.append(g_out[tuple(sl)])
            idx += size
        return grads

def concat(tensors, axis: int = 0):
    """mini.concat([a, b, c], axis=...) → Tensor"""
    return Concat.apply(*tensors, axis=axis)

# ----------------------------------------------------------------------
# Stack : np.stack wrapper ---------------------------------------------
# ----------------------------------------------------------------------
class Stack(Function):
    @staticmethod
    def apply(*tensors, axis: int = 0):
        data_list = [t.data for t in tensors]
        out_data = np.stack(data_list, axis=axis)
        requires_grad = any(t.requires_grad for t in tensors)
        out = mini.Tensor(out_data, requires_grad=requires_grad)

        ctx = Stack(*tensors)
        ctx.axis = axis
        out._ctx = ctx
        return out

    def backward(self, g_out):
        axis = self.axis
        return [np.take(g_out, i, axis=axis) for i in range(len(self.parents))]

def stack(tensors, axis: int = 0):
    """mini.stack([a, b, c], axis=...) → Tensor"""
    return Stack.apply(*tensors, axis=axis)

# ----------------------------------------
class Gather(Function):
    @staticmethod
    def apply(weight, indices):
        out = mini.Tensor(weight.data[indices], requires_grad=weight.requires_grad)
        ctx = Gather(weight)
        ctx.save_for_backward(indices, weight.data.shape)
        out._ctx = ctx
        return out
    def backward(self, g_out):
        indices, w_shape = self.saved_tensors
        grad_w = np.zeros(w_shape, dtype=g_out.dtype)
        np.add.at(grad_w, indices, g_out)   # scatter-add
        return [grad_w, None]               # indices are non-diff

#############################
# Attention utilities        #
#############################

class ScaledDotProductAttention(Function):
    """Scaled dot-product attention.

    This implementation only supports self-attention and optional causal masking
    (where positions cannot attend to subsequent positions).
    Q, K, V are expected to have shape ``(B, T, D)`` where ``B`` is the batch
    size, ``T`` the sequence length and ``D`` the embedding dimension.
    """

    @staticmethod
    def apply(q, k, v, causal: bool = False):
        d_k = q.data.shape[-1]

        # compute attention scores
        scores = np.matmul(q.data, k.data.transpose(0, 2, 1)) / np.sqrt(d_k)

        mask = None
        if causal:
            t = q.data.shape[-2]
            mask = np.triu(np.ones((t, t), dtype=bool), k=1)
            scores = np.where(mask, -1e9, scores)

        # softmax over the key dimension
        shifted = scores - scores.max(axis=-1, keepdims=True)
        exp_scores = np.exp(shifted)
        weights = exp_scores / exp_scores.sum(axis=-1, keepdims=True)

        out_data = np.matmul(weights, v.data)

        requires_grad = q.requires_grad or k.requires_grad or v.requires_grad
        out = mini.Tensor(out_data, requires_grad=requires_grad)

        ctx = ScaledDotProductAttention(q, k, v)
        ctx.causal = causal
        ctx.d_k = d_k
        ctx.mask = mask
        ctx.save_for_backward(q, k, v, mini.Tensor(weights))
        out._ctx = ctx
        return out

    def backward(self, grad_out):
        q, k, v, w = self.saved_tensors
        w_data = w.data

        # gradient w.r.t. value
        grad_v = np.matmul(w_data.transpose(0, 2, 1), grad_out) if v.requires_grad else None

        # gradients through softmax
        grad_w = np.matmul(grad_out, v.data.transpose(0, 2, 1))
        dot = (grad_w * w_data).sum(axis=-1, keepdims=True)
        grad_scores = w_data * (grad_w - dot)

        if self.causal and self.mask is not None:
            grad_scores = np.where(self.mask, 0.0, grad_scores)

        scale = 1.0 / np.sqrt(self.d_k)
        grad_q = np.matmul(grad_scores, k.data) * scale if q.requires_grad else None
        grad_k = np.matmul(grad_scores.transpose(0, 2, 1), q.data) * scale if k.requires_grad else None

        return [grad_q, grad_k, grad_v]

def scaled_dot_product_attention(q, k, v, causal: bool = False):
    """mini.scaled_dot_product_attention(q, k, v, causal=False) -> Tensor"""
    return ScaledDotProductAttention.apply(q, k, v, causal=causal)
