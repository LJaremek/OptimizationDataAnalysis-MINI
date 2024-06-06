import numpy as np


class MultiLayerPerceptron:
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
