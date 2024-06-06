"""
Co zakładamy:
* musimy napisać sieć binarną (-1, 1) do kategoryzacji obrazków: dane zdjęciowe
* musimy przetestować sieć na 3 datasetach tak jak w paperze
* Porównamy 6 testów - (ADAM, ADAMAX) x (3 data sety)
* Użyjemy samplera do wyrównania datasetów
* Dowieziemy śliczne repo i przepiękne sprawko
* Deterministic Binarisation (not stochastic)
* Na MNIST: MLP
* Na SVHN i CIFAR10: CONV-NET

Nasze pytania chętne:
* Architekura sieci

Nasze pytania niechętne:
* Kod w Pythonie 2.7, czy mamy go używać? Czy tworzyć od zera?
* Czy mamy przeprowadzać analizę matematyczną
"""
import numpy as np
import struct

from bnn import MultiLayerPerceptron


def load_mnist_images(file_path: str) -> np.array:
    with open(file_path, "rb") as f:
        # Read magic number and dimensions
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        # Read image data
        images = np.fromfile(f, dtype=np.uint8).reshape(num, rows * cols)
    return images


def load_mnist_labels(file_path: str) -> np.array:
    with open(file_path, "rb") as f:
        # Read magic number and number of items
        magic, num = struct.unpack(">II", f.read(8))
        # Read label data
        labels = np.fromfile(f, dtype=np.uint8)
    return labels


# File paths
train_images_path = "data/train-images.idx3-ubyte"
train_labels_path = "data/train-labels.idx1-ubyte"
test_images_path = "data/t10k-images.idx3-ubyte"
test_labels_path = "data/t10k-labels.idx1-ubyte"

# Load datasets
train_images = load_mnist_images(train_images_path)
train_labels = load_mnist_labels(train_labels_path)
test_images = load_mnist_images(test_images_path)
test_labels = load_mnist_labels(test_labels_path)

# Normalize images to binary values (-1 or 1)
train_images = np.where(train_images > 127, 1, -1).astype(np.float32)
test_images = np.where(test_images > 127, 1, -1).astype(np.float32)

# One-hot encode labels
num_classes = 10
train_labels_one_hot = np.eye(num_classes)[train_labels]
test_labels_one_hot = np.eye(num_classes)[test_labels]


def decode_output(output: np.array) -> int:
    return np.argmax(output, axis=1)


def calculate_accuracy(predictions, labels):
    return np.mean(predictions == labels)


# Example usage of binary neural network
if __name__ == "__main__":
    bnn = MultiLayerPerceptron(
        input_size=28*28,
        hidden_size=128,
        output_size=num_classes,
        learning_rate=0.001
        )

    bnn.train(train_images, train_labels_one_hot, epochs=1_000)

    # Test the trained BNN
    test_predictions = []
    for i in range(len(test_images)):
        test_sample = test_images[i].reshape(1, -1)  # Ensure test_sample is 2D
        predicted_output = bnn.forward(test_sample)
        predicted_class = decode_output(predicted_output)
        test_predictions.append(predicted_class[0])

    test_predictions = np.array(test_predictions)
    accuracy = calculate_accuracy(test_predictions, test_labels)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
