import numpy as np
import minigrad.tensor as mini

class AIPParameter(mini.Tensor):
    def __init__(self, data, requires_grad=True, _ctx=None):
        super().__init__(data, requires_grad=requires_grad, _ctx=_ctx)

class AIPModule:
    def __init__(self):
        self._parameters = {}
        self._modules = {}

    def __setattr__(self, name, value):
        if isinstance(value, AIPParameter):
            self._parameters[name] = value
        elif isinstance(value, AIPModule):
            self._modules[name] = value
        super().__setattr__(name, value)

    def parameters(self):
        for param in self._parameters.values():
            yield param
        for module in self._modules.values():
            yield from module.parameters()

    def named_parameters(self, prefix=''):
        for name, param in self._parameters.items():
            yield prefix + name, param
        for module_name, module in self._modules.items():
            sub_prefix = f"{prefix}{module_name}."
            yield from module.named_parameters(prefix=sub_prefix)

    def zero_grad(self):
        for param in self._parameters.values():
            param.zero_grad()
        for module in self._modules.values():
            module.zero_grad()
