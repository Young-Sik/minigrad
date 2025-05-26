import numpy as np
import urllib.request
import tarfile
import pickle
import os


class CIFAR10Loader:
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"

    def __init__(self, batch_size=32, train=True, shuffle=True):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.train = train

        self.data_dir = "./cifar-10-batches-py"
        
        if not os.path.exists(self.data_dir):
            self.download_and_extract()

        self.images, self.labels = self.load_data()
        self.num_samples = self.images.shape[0]

    def download_and_extract(self):
        file_path, _ = urllib.request.urlretrieve(self.url, "cifar-10-python.tar.gz")
        with tarfile.open(file_path, "r:gz") as tar:
            tar.extractall(path=".")

    def load_data(self):
        def load_batch(file):
            with open(file, 'rb') as fo:
                batch = pickle.load(fo, encoding='bytes')
                images = batch[b'data']
                labels = batch[b'labels']
                return images, labels

        if self.train:
            images_list, labels_list = [], []
            for i in range(1, 6):
                batch_file = os.path.join(self.data_dir, f"data_batch_{i}")
                batch_images, batch_labels = load_batch(batch_file)
                images_list.append(batch_images)
                labels_list.extend(batch_labels)
            images = np.concatenate(images_list)
            labels = np.array(labels_list)
        else:
            images, labels = load_batch(os.path.join(self.data_dir, "test_batch"))
            labels = np.array(labels)

        images = images.reshape(-1, 3, 32, 32).astype('float32') / 255.0
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
    train_loader = CIFAR10Loader(batch_size=64, train=True)

    for batch_images, batch_labels in train_loader:
        print(f"Batch shape: {batch_images.shape}, Labels shape: {batch_labels.shape}")
        break
