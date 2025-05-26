import numpy as np
from minigrad import Tensor, AIPLinear, AIPModule
class SimpleMLP(AIPModule):
    def __init__(self):
        super().__init__()
        self.fc1 = AIPLinear(2, 4)
        self.fc2 = AIPLinear(4, 1)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x).relu()
        return self.fc2(x)

    __call__ = forward

def main():
    model = SimpleMLP()
    # Dummy input and target
    x = Tensor([[1.0, 2.0]], requires_grad=True)
    target = Tensor([[1.0]])

    # Forward pass
    out = model(x)
    loss = ((out - target) ** 2).mean()
    loss.backward()

    print("Loss:", loss)
    for name, param in model.named_parameters():
        print(name, param.grad)

if __name__ == "__main__":
    main()
