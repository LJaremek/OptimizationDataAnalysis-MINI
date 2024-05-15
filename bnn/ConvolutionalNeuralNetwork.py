import numpy as np
import struct
import time


def load_mnist_images(file_path: str) -> np.array:
    with open(file_path, "rb") as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        images = np.fromfile(f, dtype=np.uint8).reshape(num, 1, rows, cols)
    return images


def load_mnist_labels(file_path: str) -> np.array:
    with open(file_path, "rb") as f:
        magic, num = struct.unpack(">II", f.read(8))
        labels = np.fromfile(f, dtype=np.uint8)
    return labels


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


def binary_sign(x: np.array) -> np.array:
    return np.where(x >= 0, 1, -1)


class ConvolutionalNeuralNetwork:
    def __init__(self, learning_rate: float = 0.01) -> None:
        self.learning_rate = learning_rate
        self.filter_size = 3
        self.num_filters = 8
        self.conv_stride = 1
        self.pool_size = 2
        self.pool_stride = 2
        self.conv_output_size = (28 - self.filter_size + 1) // self.conv_stride
        self.pool_output_size = self.conv_output_size // self.pool_size

        self.weights_conv = np.random.randn(
            self.num_filters, 1, self.filter_size, self.filter_size
            ) * np.sqrt(2. / self.filter_size**2)

        self.bias_conv = np.zeros(self.num_filters)

        _tmp = self.num_filters * self.pool_output_size * self.pool_output_size
        self.weights_fc = (
            np.random.randn(_tmp, 10) *
            np.sqrt(2. / (self.num_filters * self.pool_output_size**2))
        )

        self.bias_fc = np.zeros(10)

    def forward(self, X: np.array):
        self.X = X
        self.conv_output = self.convolve(X)
        self.pool_output = self.max_pool(self.conv_output)
        self.fc_input = self.pool_output.reshape(X.shape[0], -1)

        self.fc_output = sigmoid(
            np.dot(self.fc_input, self.weights_fc) + self.bias_fc
            )

        return self.fc_output

    def convolve(self, X: np.array) -> np.array:
        batch_size, in_channels, in_height, in_width = X.shape

        conv_output = np.zeros((
            batch_size,
            self.num_filters,
            self.conv_output_size,
            self.conv_output_size
            ))

        for f in range(self.num_filters):
            for i in range(self.conv_output_size):
                for j in range(self.conv_output_size):
                    index_i_1 = i * self.conv_stride
                    index_i_2 = i * self.conv_stride + self.filter_size
                    index_j_1 = j * self.conv_stride
                    index_j_2 = j * self.conv_stride + self.filter_size
                    region = X[:, :, index_i_1:index_i_2, index_j_1:index_j_2]

                    conv_output[:, f, i, j] = np.sum(
                        region * binary_sign(self.weights_conv[f]),
                        axis=(1, 2, 3)
                        ) + self.bias_conv[f]

        return conv_output

    def max_pool(self, X: np.array) -> np.array:
        batch_size, num_filters, in_height, in_width = X.shape

        pool_output = np.zeros((
            batch_size,
            num_filters,
            self.pool_output_size,
            self.pool_output_size
            ))

        for i in range(self.pool_output_size):
            for j in range(self.pool_output_size):
                index_i_1 = i * self.pool_stride
                index_i_2 = i * self.pool_stride + self.pool_size
                index_j_1 = j * self.pool_stride
                index_j_2 = j * self.pool_stride + self.pool_size

                region = X[:, :, index_i_1:index_i_2, index_j_1:index_j_2]
                pool_output[:, :, i, j] = np.max(region, axis=(2, 3))

        return pool_output

    def backward(self, y_true: np.array, y_pred: np.array) -> None:
        batch_size = y_true.shape[0]
        fc_error = y_pred - y_true
        fc_delta = fc_error * sigmoid_derivative(y_pred)

        self.bias_fc -= (
            self.learning_rate * np.sum(fc_delta, axis=0) / batch_size
        )

        self.weights_fc -= (
            self.learning_rate * np.dot(self.fc_input.T, fc_delta) / batch_size
        )

        pool_error = np.dot(
            fc_delta, self.weights_fc.T
            ).reshape(self.pool_output.shape)

        conv_error = np.zeros(self.conv_output.shape)
        for f in range(self.num_filters):
            for i in range(self.pool_output_size):
                for j in range(self.pool_output_size):
                    index_i_1 = i * self.pool_stride
                    index_i_2 = i * self.pool_stride + self.pool_size
                    index_j_1 = j * self.pool_stride
                    index_j_2 = j * self.pool_stride + self.pool_size

                    region = self.conv_output[
                        :,
                        f,
                        index_i_1:index_i_2,
                        index_j_1:index_j_2
                        ]

                    max_val = np.max(region, axis=(1, 2), keepdims=True)

                    index_i_1 = i * self.pool_stride
                    index_i_2 = i * self.pool_stride + self.pool_size
                    index_j_1 = j * self.pool_stride
                    index_j_2 = j * self.pool_stride + self.pool_size

                    new_value = (
                        (region == max_val) *
                        pool_error[:, f, i, j][:, None, None]
                    )

                    conv_error[
                        :,
                        f,
                        index_i_1:index_i_2,
                        index_j_1:index_j_2
                        ] += new_value

        self.bias_conv -= (
            self.learning_rate *
            np.sum(conv_error, axis=(0, 2, 3)) /
            batch_size
        )

        for f in range(self.num_filters):
            for i in range(self.conv_output_size):
                for j in range(self.conv_output_size):
                    index_i_1 = i * self.conv_stride
                    index_i_2 = i * self.conv_stride + self.filter_size
                    index_j_1 = j * self.conv_stride
                    index_j_2 = j * self.conv_stride + self.filter_size

                    region = self.X[
                        :,
                        :,
                        index_i_1:index_i_2,
                        index_j_1:index_j_2
                        ]

                    self.weights_conv[f] -= (
                        self.learning_rate *
                        np.sum(
                            conv_error[:, f, i, j][:, None, None, None] *
                            binary_sign(region), axis=0
                            ) /
                        batch_size
                    )

    def train(
            self,
            X: np.array,
            y: np.array,
            epochs: int = 10,
            batch_size: int = 32
            ) -> None:

        for epoch in range(epochs):
            perm = np.random.permutation(len(X))
            X = X[perm]
            y = y[perm]
            for i in range(0, len(X), batch_size):
                X_batch = X[i:i+batch_size]
                y_batch = y[i:i+batch_size]
                y_pred = self.forward(X_batch)
                y_true = np.zeros((y_batch.shape[0], 10))
                y_true[np.arange(y_batch.shape[0]), y_batch] = 1
                self.backward(y_true, y_pred)
            print(f"Epoch {epoch + 1}/{epochs} completed")


def main() -> None:
    train_images_path = "data/train-images.idx3-ubyte"
    train_labels_path = "data/train-labels.idx1-ubyte"
    test_images_path = "data/t10k-images.idx3-ubyte"
    test_labels_path = "data/t10k-labels.idx1-ubyte"

    train_images = load_mnist_images(train_images_path)
    train_labels = load_mnist_labels(train_labels_path)
    test_images = load_mnist_images(test_images_path)
    test_labels = load_mnist_labels(test_labels_path)

    # Normalize images to binary values (-1 or 1)
    train_images = np.where(train_images > 127, 1, -1).astype(np.float32)
    test_images = np.where(test_images > 127, 1, -1).astype(np.float32)

    model = ConvolutionalNeuralNetwork(learning_rate=0.01)

    start_time = time.time()
    print("Start training...")
    model.train(train_images, train_labels, epochs=10, batch_size=64)
    print("End training.")
    print(f"Training time: {time.time() - start_time:.2f} seconds")

    correct = 0
    for i in range(len(test_images)):
        y_pred = model.forward(test_images[i:i+1])
        if np.argmax(y_pred) == test_labels[i]:
            correct += 1

    accuracy = correct / len(test_images)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")


if __name__ == "__main__":
    main()
