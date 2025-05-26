# Linear layer mapping from 3 features to 2
layer = AIPLinear(3, 2)

# Example input batch (1 x 3)
x = Tensor([[1.0, 2.0, 3.0]], requires_grad=True)

# Forward pass
out = layer(x)

# Loss is the sum of outputs
loss = out.sum()
loss.backward()

print("Output:", out)
print("Weight grad:", layer.weight.grad)
if layer.use_bias:
    print("Bias grad:", layer.bias.grad)
