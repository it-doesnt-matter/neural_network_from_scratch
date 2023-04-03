from abc import ABC, abstractmethod

import numpy as np


class Accuracy(ABC):
    def __init__(self) -> None:
        self.accumulated_sum = 0
        self.accumulated_count = 0

    def calculate_step_accuracy(self, prediction: np.ndarray, target: np.ndarray) -> float:
        comparisons = self.compare(prediction, target)
        step_accuracy = np.mean(comparisons)
        self.accumulated_sum += np.sum(comparisons)
        self.accumulated_count += len(comparisons)
        return step_accuracy

    def calculate_epoch_accuracy(self) -> float:
        epoch_accuracy = self.accumulated_sum / self.accumulated_count
        self.accumulated_sum = 0
        self.accumulated_count = 0
        return epoch_accuracy

    @abstractmethod
    def compare(self, prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
        pass


class AccuracyCategorical(Accuracy):
    def compare(self, prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
        return prediction == target
