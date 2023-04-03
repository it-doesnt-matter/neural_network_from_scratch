from abc import ABC, abstractmethod

import numpy as np


class Loss(ABC):
    def __init__(self) -> None:
        self.accumulated_sum = 0
        self.accumulated_count = 0

    def calculate_step_loss(self, prediction: np.ndarray, target: np.ndarray) -> float:
        sample_losses = self.forward(prediction, target)
        step_loss = np.mean(sample_losses)
        self.accumulated_sum += np.sum(sample_losses)
        self.accumulated_count += len(sample_losses)
        return step_loss

    def calculate_epoch_loss(self) -> float:
        epoch_loss = self.accumulated_sum / self.accumulated_count
        self.accumulated_sum = 0
        self.accumulated_count = 0
        return epoch_loss

    @abstractmethod
    def forward(self, prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def backward(self, prediction: np.ndarray, target: np.ndarray) -> None:
        pass

    @abstractmethod
    def to_string(self) -> str:
        pass


class LossMSE(Loss):
    def forward(self, prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
        return np.mean((target - prediction) ** 2, axis=-1)

    def backward(self, prediction: np.ndarray, target: np.ndarray) -> None:
        samples = len(prediction)
        outputs = len(prediction[0])
        self.d_inputs = -2 * (target - prediction) / outputs
        self.d_inputs = self.d_inputs / samples

    def to_string(self) -> str:
        return "Mean Squared Error"


class LossCategoricalCrossEntropy(Loss):
    def forward(self, prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
        # this prevents division by zero
        clipped_prediction = np.clip(prediction, 1e-7, 1 - 1e-7)
        correct_prediction = clipped_prediction[range(len(prediction)), target]
        return -np.log(correct_prediction)

    def backward(self, d_values: np.ndarray, target: np.ndarray) -> None:
        d_values[d_values == 0] = np.finfo(np.float64).tiny
        # this turns the target into a one-hot vector
        target = np.eye(len(d_values[0]))[target]
        self.d_inputs = -target / d_values
        self.d_inputs = self.d_inputs / len(d_values)

    def to_string(self) -> str:
        return "Categorical Cross-Entropy"
