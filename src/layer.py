from abc import ABC, abstractmethod

import numpy as np


class Layer(ABC):
    @abstractmethod
    def forward(self, inpt: np.ndarray) -> None:
        pass

    @abstractmethod
    def backward(self, d_value: np.ndarray) -> None:
        pass

    @property
    @abstractmethod
    def trainable(self) -> bool:
        pass

    @property
    @abstractmethod
    def info(self) -> tuple[str, int]:
        pass


class LayerInput(Layer):
    def __init__(self, amount_neurons: int) -> None:
        self.size = amount_neurons

    def forward(self, inpt: np.ndarray) -> None:
        self.output = inpt

    def backward(self, d_values: np.ndarray) -> None:
        None

    @property
    def trainable(self) -> bool:
        return False

    @property
    def info(self) -> tuple[str, int]:
        return ("Input", self.size)


class LayerDense(Layer):
    def __init__(self, amount_inputs: int, amount_neurons: int) -> None:
        rng = np.random.default_rng()
        self.weights = 0.01 * rng.standard_normal((amount_inputs, amount_neurons))
        self.biases = np.zeros((1, amount_neurons))
        self.size = amount_neurons

    def forward(self, inpt: np.ndarray) -> None:
        self.inpt = inpt
        self.output = np.matmul(inpt, self.weights) + self.biases

    def backward(self, d_values: np.ndarray) -> None:
        self.d_weights = np.matmul(self.inpt.T, d_values)
        self.d_biases = np.sum(d_values, axis=0, keepdims=True)
        self.d_inputs = np.matmul(d_values, self.weights.T)

    @property
    def trainable(self) -> bool:
        return True

    @property
    def info(self) -> tuple[str, int]:
        return ("Dense", self.size)


class LayerDropout(Layer):
    # rate indicates how many neurons should be DISABLED
    def __init__(self, rate: float) -> None:
        self.rate = 1 - rate

    def forward(self, inpt: np.ndarray) -> None:
        self.inpt = inpt
        rng = np.random.default_rng()
        self.binary_mask = rng.binomial(1, self.rate, size=inpt.shape) / self.rate
        self.output = inpt * self.binary_mask

    def forward_without_dropout(self, inpt: np.ndarray) -> None:
        self.inpt = inpt
        self.output = inpt.copy()

    @property
    def trainable(self) -> bool:
        return False

    # this needs a review
    @property
    def info(self) -> tuple[str, int]:
        return ("Dropout", self.size)


class Activation(Layer, ABC):
    @abstractmethod
    def forward(self, inpt: np.ndarray) -> None:
        pass

    @abstractmethod
    def backward(self, d_values: np.ndarray) -> None:
        pass

    @abstractmethod
    def predict(self, output: np.ndarray) -> np.ndarray:
        pass

    @property
    @abstractmethod
    def trainable(self) -> bool:
        pass

    @property
    @abstractmethod
    def info(self) -> tuple[str, int]:
        pass


class ActivationSigmoid(Layer):
    def __init__(self, amount_inputs: int, amount_neurons: int) -> None:
        self.size = amount_neurons

    def forward(self, inpt: np.ndarray) -> None:
        self.inpt = inpt
        self.output = 1 / (1 + np.exp(-inpt))

    def backward(self, d_values: np.ndarray) -> None:
        self.d_inputs = d_values * (1 - self.output) * self.output

    def predict(self, output: np.ndarray) -> np.ndarray:
        return (output > 0.5) * 1

    @property
    def trainable(self) -> bool:
        return False

    @property
    def info(self) -> tuple[str, int]:
        return ("Sigmoid", self.size)


class ActivationReLU(Layer):
    def __init__(self, amount_inputs: int, amount_neurons: int) -> None:
        self.size = amount_neurons

    def forward(self, inpt: np.ndarray) -> None:
        self.inpt = inpt
        self.output = np.maximum(0, inpt)

    def backward(self, d_values: np.ndarray) -> None:
        self.d_inputs = d_values.copy()
        self.d_inputs[self.inpt <= 0] = 0

    def predict(self, output: np.ndarray) -> np.ndarray:
        return output

    @property
    def trainable(self) -> bool:
        return False

    @property
    def info(self) -> tuple[str, int]:
        return ("ReLU", self.size)


class ActivationSoftmax(Layer):
    def __init__(self, amount_inputs: int, amount_neurons: int) -> None:
        self.size = amount_neurons

    def forward(self, inpt: np.ndarray) -> None:
        self.inpt = inpt

        # this prevents overflow when using np.exp
        # due to normalization, the input stays the same
        shifted_input = inpt - np.max(inpt, axis=1, keepdims=True)

        self.output = np.exp(shifted_input) / np.sum(np.exp(shifted_input), axis=1, keepdims=True)

    def backward(self, d_values: np.ndarray) -> None:
        self.d_inputs = np.empty_like(d_values)
        zipped = zip(self.output, d_values, strict=True)
        for index, (single_output, single_d_values) in enumerate(zipped):
            single_output = single_output.reshape(-1, 1)
            jacobian_matrix = (np.diagflat(single_output)
                               - np.matmul(single_output, single_output.T))
            self.d_inputs[index] = np.matmul(jacobian_matrix, single_d_values)

    def predict(self, output: np.ndarray) -> np.ndarray:
        return np.argmax(output, axis=1)

    @property
    def trainable(self) -> bool:
        return False

    @property
    def info(self) -> tuple[str, int]:
        return ("Softmax", self.size)
