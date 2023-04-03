import numpy as np
from tqdm.rich import trange

import utils
from accuracy import Accuracy
from layer import Layer, LayerDropout
from loss import Loss
from optimizer import Optimizer


class SequentialNetwork:
    def __init__(
        self,
        optimizer: Optimizer,
        accuracy: Accuracy,
        loss: Loss,
        layers: list[tuple[type[Layer], int]],
    ) -> None:
        self.optimizer = optimizer
        self.accuracy = accuracy
        self.loss = loss
        layer, inputs = layers[0]
        self.input_layer = layer(inputs)
        self.layers = []
        for (_, inputs, *_), (layer, neurons) in zip(layers[:-1], layers[1:], strict=True):
            self.layers.append(layer(inputs, neurons))

    def get_architecture(self) -> list[tuple[Layer, int]]:
        name, size = self.input_layer.info
        result = [(name, size)]
        for layer in self.layers:
            name, size = layer.info
            result.append((name, size))
        return result

    def train(
        self,
        inpt: np.ndarray,
        target: np.ndarray,
        epochs: int,
        batch_size: int,
        validation_input: np.ndarray = None,
        validation_target: np.ndarray = None,
    ) -> dict[str, list[float]]:
        if len(inpt) % batch_size == 0:
            steps = len(inpt) // batch_size
        else:
            steps = (len(inpt) // batch_size) + 1

        summary = {
            "epochs": epochs,
            "batch_size": batch_size,
            "training_loss": [],
            "validation_loss": [],
            "training_accuracy": [],
            "validation_accuracy": [],
            "learning_rate": [],
        }

        for epoch in trange(1, epochs + 1):
            inpt, target = utils.shuffle_data(inpt, target)
            for step in range(steps):
                batch_input = inpt[step * batch_size:(step + 1) * batch_size]
                batch_target = target[step * batch_size:(step + 1) * batch_size]

                output = self._forward(batch_input, training=True)
                self.loss.calculate_step_loss(output, batch_target)

                prediction = self.layers[-1].predict(output)
                self.accuracy.calculate_step_accuracy(prediction, batch_target)

                self._backward(output, batch_target)

                self.optimizer.update_params(self.layers, epoch)

            summary["training_loss"].append(self.loss.calculate_epoch_loss())
            summary["training_accuracy"].append(self.accuracy.calculate_epoch_accuracy())
            summary["learning_rate"].append(self.optimizer.current_lr)

            if validation_input is not None and validation_target is not None:
                loss, acc = self.evaluate(validation_input, validation_target)
                summary["validation_loss"].append(loss)
                summary["validation_accuracy"].append(acc)

        return summary

    def predict(self, inpt: np.ndarray, batch_size: int) -> np.ndarray:
        if len(inpt) % batch_size == 0:
            steps = len(inpt) // batch_size
        else:
            steps = (len(inpt) // batch_size) + 1

        output = []
        for step in range(steps):
            batch_input = inpt[step * batch_size:(step + 1) * batch_size]
            batch_output = self._forward(batch_input, training=False)
            output.append(batch_output)
        return self.layers[-1].predict(np.vstack(output))

    def evaluate(self, inpt: np.ndarray, target: np.ndarray) -> tuple[float, float]:
        output = self._forward(inpt, training=False)
        self.loss.calculate_step_loss(output, target)

        prediction = self.layers[-1].predict(output)
        self.accuracy.calculate_step_accuracy(prediction, target)

        # they return the same values as the step methods
        # but they need to be called to reset the "accumulated_..." attributes
        loss = self.loss.calculate_epoch_loss()
        accuracy = self.accuracy.calculate_epoch_accuracy()

        return loss, accuracy

    def _forward(self, inpt: np.ndarray, training: bool) -> np.ndarray:
        self.input_layer.forward(inpt)
        self.layers[0].forward(self.input_layer.output)
        for previous, current in zip(self.layers[:-1], self.layers[1:], strict=True):
            if not training and isinstance(current, LayerDropout):
                # don't disable any layers during prediction
                current.forward_without_dropout(previous.output)
            else:
                current.forward(previous.output)
        return self.layers[-1].output

    def _backward(self, output: np.ndarray, target: np.ndarray) -> None:
        self.loss.backward(output, target)
        self.layers[-1].backward(self.loss.d_inputs)
        zipped = zip(reversed(self.layers[:-1]), reversed(self.layers[1:]), strict=True)
        for current, previous in zipped:
            current.backward(previous.d_inputs)

    def get_params(self) -> list[tuple[np.ndarray, np.ndarray]]:
        return [(layer.weights, layer.biases) for layer in self.layers if layer.trainable]

    def set_params(self, params: list[tuple[np.ndarray, np.ndarray]]) -> None:
        for layer in self.layers:
            if layer.trainable:
                popped = params[0]
                params = params[1:]
                layer.weights = popped[0]
                layer.biases = popped[1]

    def _get_flattened_params(self) -> tuple[list[float], list[any]]:
        params = self.get_params()
        structure = []
        for param in params:
            structure.append((param[0].shape, param[1].shape))
        params = self._flatten(params)
        return params, structure

    def _get_gradient(self) -> list[tuple[np.ndarray, np.ndarray]]:
        return [(layer.d_weights, layer.d_biases) for layer in self.layers if layer.trainable]

    def _get_flattened_gradient(self) -> tuple[list[float], list[any]]:
        gradient = self._get_gradient()
        gradient = self._flatten(gradient)
        return gradient

    def _flatten(self, xs: list[any] | tuple[any] | np.ndarray) -> list[any]:
        result = []
        if isinstance(xs, list | tuple | np.ndarray):
            for x in xs:
                result.extend(self._flatten(x))
        else:
            result.append(xs)
        return result

    def _revert_flatten(
        self,
        xs: list[float],
        structure: list[any],
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        result = []
        index = 0
        for neuron in structure:
            amount_weights = neuron[0][0] * neuron[0][1]
            weights = xs[index:index + amount_weights]
            weights = np.reshape(weights, (neuron[0][0], neuron[0][1]))
            index += amount_weights

            amount_biases = neuron[1][0] * neuron[1][1]
            biases = xs[index:index + amount_biases]
            biases = np.reshape(biases, (neuron[1][0], neuron[1][1]))
            index += amount_biases

            result.append((weights, biases))

        return result

    # this only checks the backpropagation
    # the correctness of forward() is assumed
    # this should not be used with dropout
    def check_gradient(self, inpt: np.ndarray, target: np.ndarray) -> None:
        EPSILON = 1e-4
        ACCEPTANCE_THRESHOLD = 1e-5
        old_params = self.get_params()
        params, structure = self._get_flattened_params()
        grad_approximation = np.zeros(len(params))

        # calculation of the gradient via backpropagation
        output = self._forward(inpt, training=False)
        self.loss.calculate_step_loss(output, target)
        self._backward(output, target)
        self.optimizer.update_params(self.layers, 1)
        grad = self._get_flattened_gradient()
        grad = np.array(grad)

        # calculation of the gradient vie numerical differentiation
        for i in range(len(params)):
            param_plus = np.copy(params)
            param_plus[i] += EPSILON
            param_plus = self._revert_flatten(param_plus, structure)
            self.set_params(param_plus)
            output = self._forward(inpt, training=False)
            plus_loss = self.loss.calculate_step_loss(output, target)

            param_minus = np.copy(params)
            param_minus[i] -= EPSILON
            param_minus = self._revert_flatten(param_minus, structure)
            self.set_params(param_minus)
            output = self._forward(inpt, training=False)
            minus_loss = self.loss.calculate_step_loss(output, target)

            grad_approximation[i] = (plus_loss - minus_loss) / (2 * EPSILON)

        # evaluation of the difference between the two gradients
        numerator = np.linalg.norm(grad - grad_approximation)
        denominator = np.linalg.norm(grad) + np.linalg.norm(grad_approximation)
        difference = numerator / denominator

        print(f"The difference is {difference}, ", end="")
        if difference > ACCEPTANCE_THRESHOLD:
            print("that's too high")
        else:
            print("that seems fine")

        self.set_params(old_params)
