import numpy as np
import urllib.request
import gzip
import os


class MNISTLoader:
    urls = {
        "train_images": "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz",
        "train_labels": "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz",
        "test_images": "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz",
        "test_labels": "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz"
    }

    def __init__(self, batch_size=32, train=True, shuffle=True):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.train = train

        self.data_dir = "./mnist-data"
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            self.download_data()

        self.images, self.labels = self.load_data()
        self.num_samples = self.images.shape[0]

    def download_data(self):
        for key, url in self.urls.items():
            filename = os.path.join(self.data_dir, url.split('/')[-1])
            urllib.request.urlretrieve(url, filename)

    def load_data(self):
        def load_images(filename):
            with gzip.open(filename, 'rb') as f:
                f.read(16)  # skip header
                data = np.frombuffer(f.read(), dtype=np.uint8).reshape(-1, 28, 28)
                return data.astype('float32') / 255.0

        def load_labels(filename):
            with gzip.open(filename, 'rb') as f:
                f.read(8)  # skip header
                return np.frombuffer(f.read(), dtype=np.uint8)

        if self.train:
            images = load_images(os.path.join(self.data_dir, 'train-images-idx3-ubyte.gz'))
            labels = load_labels(os.path.join(self.data_dir, 'train-labels-idx1-ubyte.gz'))
        else:
            images = load_images(os.path.join(self.data_dir, 't10k-images-idx3-ubyte.gz'))
            labels = load_labels(os.path.join(self.data_dir, 't10k-labels-idx1-ubyte.gz'))

        return images, labels

    def __iter__(self):
        indices = np.arange(self.num_samples)
        if self.shuffle:
            np.random.shuffle(indices)

        for start_idx in range(0, self.num_samples, self.batch_size):
            end_idx = start_idx + self.batch_size
            batch_indices = indices[start_idx:end_idx]
            yield self.images[batch_indices], self.labels[batch_indices]


# Example usage:
if __name__ == '__main__':
    train_loader = MNISTLoader(batch_size=64, train=True)

    for batch_images, batch_labels in train_loader:
        print(f"Batch shape: {batch_images.shape}, Labels shape: {batch_labels.shape}")
        break
