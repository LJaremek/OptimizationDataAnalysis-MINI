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


class BinaryNeuralNetwork:
    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            output_size: int,
            learning_rate: float = 0.01
            ) -> None:

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        # Initialize weights with He initialization
        self.weights_input_hidden = (
            np.random.randn(input_size, hidden_size) *
            np.sqrt(2. / input_size)
        )

        self.weights_hidden_output = (
            np.random.randn(hidden_size, output_size) *
            np.sqrt(2. / hidden_size)
        )

        self.bias_hidden = np.zeros(hidden_size)
        self.bias_output = np.zeros(output_size)

        # Adam parameters
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.epsilon = 1e-8
        self.m_w_ih = np.zeros_like(self.weights_input_hidden)
        self.v_w_ih = np.zeros_like(self.weights_input_hidden)
        self.m_w_ho = np.zeros_like(self.weights_hidden_output)
        self.v_w_ho = np.zeros_like(self.weights_hidden_output)
        self.m_b_h = np.zeros_like(self.bias_hidden)
        self.v_b_h = np.zeros_like(self.bias_hidden)
        self.m_b_o = np.zeros_like(self.bias_output)
        self.v_b_o = np.zeros_like(self.bias_output)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self, X):
        self.hidden_input = (
            np.dot(X, self.weights_input_hidden) +
            self.bias_hidden
        )

        self.hidden_output = self.sigmoid(self.hidden_input)
        self.final_input = (
            np.dot(self.hidden_output, self.weights_hidden_output) +
            self.bias_output
        )

        self.final_output = self.sigmoid(self.final_input)
        return self.final_output

    def backward(self, X, y, output) -> None:
        output_error = y - output
        output_delta = output_error * self.sigmoid_derivative(output)

        hidden_error = np.dot(output_delta, self.weights_hidden_output.T)
        hidden_delta = (
            hidden_error *
            self.sigmoid_derivative(self.hidden_output)
        )

        # Adam optimization for weights and biases
        self.m_w_ho = (
            self.beta1 * self.m_w_ho +
            (1 - self.beta1) *
            np.dot(self.hidden_output.T, output_delta)
        )

        self.v_w_ho = (
            self.beta2 * self.v_w_ho +
            (1 - self.beta2) *
            (np.dot(self.hidden_output.T, output_delta) ** 2)
        )

        m_hat_w_ho = self.m_w_ho / (1 - self.beta1)
        v_hat_w_ho = self.v_w_ho / (1 - self.beta2)
        self.weights_hidden_output += (
            self.learning_rate *
            m_hat_w_ho /
            (np.sqrt(v_hat_w_ho) + self.epsilon)
        )

        self.m_b_o = (
            self.beta1 *
            self.m_b_o +
            (1 - self.beta1) *
            np.sum(output_delta, axis=0)
        )

        self.v_b_o = (
            self.beta2 *
            self.v_b_o +
            (1 - self.beta2) *
            (np.sum(output_delta, axis=0) ** 2)
        )

        m_hat_b_o = self.m_b_o / (1 - self.beta1)
        v_hat_b_o = self.v_b_o / (1 - self.beta2)
        self.bias_output += (
            self.learning_rate *
            m_hat_b_o /
            (np.sqrt(v_hat_b_o) + self.epsilon)
        )

        self.m_w_ih = (
            self.beta1 *
            self.m_w_ih +
            (1 - self.beta1) *
            np.dot(X.T, hidden_delta)
        )

        self.v_w_ih = (
            self.beta2 *
            self.v_w_ih +
            (1 - self.beta2) *
            (np.dot(X.T, hidden_delta) ** 2)
        )

        m_hat_w_ih = self.m_w_ih / (1 - self.beta1)
        v_hat_w_ih = self.v_w_ih / (1 - self.beta2)
        self.weights_input_hidden += (
            self.learning_rate *
            m_hat_w_ih /
            (np.sqrt(v_hat_w_ih) + self.epsilon)
        )

        self.m_b_h = (
            self.beta1 *
            self.m_b_h +
            (1 - self.beta1) *
            np.sum(hidden_delta, axis=0)
        )

        self.v_b_h = (
            self.beta2 *
            self.v_b_h +
            (1 - self.beta2) *
            (np.sum(hidden_delta, axis=0) ** 2)
        )

        m_hat_b_h = self.m_b_h / (1 - self.beta1)
        v_hat_b_h = self.v_b_h / (1 - self.beta2)

        self.bias_hidden += (
            self.learning_rate *
            m_hat_b_h /
            (np.sqrt(v_hat_b_h) + self.epsilon)
        )

    def train(self, X, y, epochs=100) -> None:
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output)
            if epoch % 100 == 0:
                loss = np.mean(np.square(y - output))
                print(f"Epoch {epoch}, Loss: {loss}")


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
    bnn = BinaryNeuralNetwork(
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
