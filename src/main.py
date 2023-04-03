import utils
from accuracy import AccuracyCategorical
from layer import ActivationReLU, ActivationSoftmax, LayerDense, LayerInput
from loss import LossCategoricalCrossEntropy
from network import SequentialNetwork
from optimizer import OptimizerSGD


def main() -> None:
    sgd = OptimizerSGD(0.1, decay=0.98, decay_epochs=100, min_rate=0.002)
    acc = AccuracyCategorical()
    cce = LossCategoricalCrossEntropy()
    nn = SequentialNetwork(
        sgd,
        acc,
        cce,
        [
            (LayerInput, 2),
            (LayerDense, 64),
            (ActivationReLU, 64),
            (LayerDense, 64),
            (ActivationReLU, 64),
            (LayerDense, 3),
            (ActivationSoftmax, 3),
        ],
    )

    data, data_val, target, target_val = utils.generate_spiral_data()
    summary = nn.train(
        data,
        target,
        epochs=10_000,
        batch_size=16,
        validation_input=data_val,
        validation_target=target_val,
    )
    file_name = "summaries/x.png"
    utils.draw_training_summary(
        network=nn,
        data=data,
        target=target,
        summary=summary,
        optimizer_info=nn.optimizer.get_info(),
        loss_name=nn.loss.to_string(),
        file_name=file_name,
    )


if __name__ == "__main__":
    main()
