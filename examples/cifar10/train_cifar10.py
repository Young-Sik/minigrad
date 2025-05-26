import numpy as np
from minigrad import Tensor, AIPModule, AIPLinear
from examples.cifar10.load_cifar10 import CIFAR10Loader

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

def train(model, dataloader, epochs=10, lr=0.01):

    for epoch in range(epochs):
        total_loss = 0
        for images, labels in dataloader:
            images = images.reshape(images.shape[0], -1)  # Flatten images
            inputs = Tensor(images)
            targets = Tensor(labels)

            model.zero_grad()
            outputs = model(inputs)
            loss = outputs.cross_entropy(targets)
            loss.backward()
            for name, param in model.named_parameters():
                param.data -= lr * param.grad

            total_loss += loss.data
                
        avg_loss = total_loss / dataloader.batch_size
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss}")

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
    train_loader = CIFAR10Loader(batch_size=64, train=True)

    # Train the model
    train(model, train_loader, epochs=30, lr=0.01)
 
    # save the model parameters
    model.save_parameters('cifar10_mlp_model.npz')
