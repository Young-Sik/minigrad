import numpy as np
from minigrad import Tensor, AIPModule, AIPLinear
from examples.mnist.load_mnist import MNISTLoader

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
        
if __name__ == "__main__":
    
    train_loader = MNISTLoader(batch_size=64, train=True)
    test_loader = MNISTLoader(batch_size=64, train=False)

    model = MNISTMLP()
    
    train(model, train_loader, epochs=10, lr=0.01)

    # Save the model parameters
    model.save_parameters("mnist_mlp_params.npz")
    print("Model parameters saved to mnist_mlp_params.npz")
    
