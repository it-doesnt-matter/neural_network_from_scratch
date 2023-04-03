from abc import ABC, abstractmethod

import numpy as np

from layer import Layer


class Optimizer(ABC):
    @abstractmethod
    def update_params(self, layers: list[Layer], epoch: int) -> None:
        pass

    @abstractmethod
    def get_info(self) -> dict[str, str]:
        pass


class OptimizerSGD(Optimizer):
    def __init__(
        self,
        learning_rate: float,
        decay: float = 0.0,
        decay_epochs: int = 10,
        min_rate: float = 0.0,
    ) -> None:
        self.start_lr = learning_rate
        self.current_lr = learning_rate
        self.decay = decay
        self.decay_epochs = decay_epochs
        self.min_rate = min_rate

    def update_params(self, layers: list[Layer], epoch: int) -> None:
        if self.decay != 0 and epoch % self.decay_epochs == 0:
            self.current_lr *= self.decay
            if self.min_rate != 0:
                self.current_lr = max(self.current_lr, self.min_rate)

        for layer in layers:
            if layer.trainable:
                layer.weights += -self.current_lr * layer.d_weights
                layer.biases += -self.current_lr * layer.d_biases

    def get_info(self) -> dict[str, str | float | int]:
        return {
            "name": "SGD",
            "initial learning rate": self.start_lr,
            "decay": self.decay,
            "decay epochs": self.decay_epochs,
            "min learning rate": self.min_rate,
        }


class OptimizerAdam(Optimizer):
    def __init__(
        self,
        learning_rate: float,
        decay: float = 0.0,
        decay_epochs: int = 10,
        min_rate: float = 0.0,
        epsilon: float = 1e-7,
        beta_1: float = 0.9,
        beta_2: float = 0.999,
    ) -> None:
        self.start_lr = learning_rate
        self.current_lr = learning_rate
        self.decay = decay
        self.decay_epochs = decay_epochs
        self.min_rate = min_rate
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    def update_params(self, layers: list[Layer], epoch: int) -> None:
        if self.decay != 0 and epoch % self.decay_epochs == 0:
            self.current_lr *= self.decay
            if self.min_rate != 0:
                self.current_lr = max(self.current_lr, self.min_rate)

        for layer in layers:
            if layer.trainable:
                if not hasattr(layer, "weight_cache"):
                    layer.weight_momentums = np.zeros_like(layer.weights)
                    layer.weight_cache = np.zeros_like(layer.weights)
                    layer.bias_momentums = np.zeros_like(layer.biases)
                    layer.bias_cache = np.zeros_like(layer.biases)

                layer.weight_momentums = (
                    self.beta_1 * layer.weight_momentums + (1 - self.beta_1) * layer.d_weights
                )
                layer.bias_momentums = (
                    self.beta_1 * layer.bias_momentums + (1 - self.beta_1) * layer.d_biases
                )

                corrected_weight_momentums = layer.weight_momentums / (1 - self.beta_1**epoch)
                corrected_bias_momentums = layer.bias_momentums / (1 - self.beta_1**epoch)

                layer.weight_cache = (
                    self.beta_2 * layer.weight_cache + (1 - self.beta_2) * layer.d_weights**2
                )
                layer.bias_cache = (
                    self.beta_2 * layer.bias_cache + (1 - self.beta_2) * layer.d_biases**2
                )

                corrected_weight_cache = layer.weight_cache / (1 - self.beta_2**epoch)
                corrected_bias_cache = layer.bias_cache / (1 - self.beta_2**epoch)

                layer.weights += (-self.current_lr * corrected_weight_momentums
                                  / (np.sqrt(corrected_weight_cache) + self.epsilon))
                layer.biases += (-self.current_lr * corrected_bias_momentums
                                 / (np.sqrt(corrected_bias_cache) + self.epsilon))

    def get_info(self) -> dict[str, str | float | int]:
        return {
            "name": "Adam",
            "initial learning rate": self.start_lr,
            "final learning rate": self.current_lr,
            "decay": self.decay,
            "decay epochs": self.decay_epochs,
            "min learning rate": self.min_rate,
            "epsilon": self.epsilon,
            "beta 1": self.beta_1,
            "beta 2": self.beta_2,
        }
