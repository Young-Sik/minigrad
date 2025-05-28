import numpy as np
from minigrad import Tensor, AIPModule, AIPLinear
from load_cifar10 import CIFAR10Loader

class CIFAR10MLP(AIPModule):
    """Simple MLP for CIFAR-10 classification."""

    def __init__(self):
        super().__init__()
        self.fc1 = AIPLinear(32 * 32 * 3, 256)
        self.fc2 = AIPLinear(256, 128)
        self.fc3 = AIPLinear(128, 10)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x).relu()
        x = self.fc2(x).relu()
        return self.fc3(x)

    __call__ = forward


def evaluate(model, dataloader):
    correct = 0
    total = 0

    for images, labels in dataloader:
        images = images.reshape(images.shape[0], -1)
        inputs = Tensor(images)
        outputs = model(inputs)
        predictions = np.argmax(outputs.data, axis=1)

        correct += (predictions == labels).sum()
        total += labels.shape[0]

    accuracy = correct / total
    print(f"Evaluation accuracy: {accuracy * 100:.2f}%")

if __name__ == '__main__':
    # Initialize the model and data loader
    model = CIFAR10MLP()
    test_loader = CIFAR10Loader(batch_size=64, train=False)

    # Load the trained model parameters if available
    model.load_parameters('cifar10_mlp_model.npz')
    
    # Evaluate the model
    evaluate(model, test_loader)
