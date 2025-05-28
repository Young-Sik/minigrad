import numpy as np
from minigrad import Tensor, AIPModule, AIPLinear
from load_mnist import MNISTLoader


class MNISTMLP(AIPModule):
    """Simple two-layer MLP for MNIST classification."""

    def __init__(self):
        super().__init__()
        self.fc1 = AIPLinear(28 * 28, 128)
        self.fc2 = AIPLinear(128, 10)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x).relu()
        return self.fc2(x)

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


if __name__ == "__main__":
    
    test_loader = MNISTLoader(batch_size=64, train=False)

    model = MNISTMLP()
    
    # Load the model parameters
    new_model = MNISTMLP()
    new_model.load_parameters("mnist_mlp_params.npz")
    print("Model parameters loaded from mnist_mlp_params.npz")
    
    # Evaluate the loaded model
    evaluate(new_model, test_loader)
