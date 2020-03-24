import os
import argparse
from src.config import config
from distutils.util import strtobool
os.environ["CUDA_VISIBLE_DEVICES"] = str(config.device_id)
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from intrinsic.fastfood import WrapFastfood
from intrinsic.dense import WrapDense
from src.models import RegularCNNModel, FCNAsInPAper, _kaiming_normal
from src.data import get_loaders
from src.utils import use_model, parameter_count
from src.utils import get_writer, add_paths_to_config, get_exponential_range
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss
from ignite.handlers import EarlyStopping

def main(config):
    """

    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_type", help="Model type", choices=["cnn", "fcn"])
    parser.add_argument("-w", "--wrap_type", help="Wrap type", choices=["fastfood", "dense"])
    parser.add_argument("--he_normal", type=strtobool, default=False, help="If he normal init")
    args = parser.parse_args()
    task_name = "MNIST"
    model_type = args.model_type
    wrap_type = args.wrap_type
    he_normal = args.he_normal

    # Sort out paths and get tensorboard write
    config = add_paths_to_config(config)

    writer = get_writer(config.tensorboard_dir)

    # Get intrinsic dimension options
    intrinsic_array = get_exponential_range(3, 14)

    # Iterate over intrinsic dimension size and
    # record highest achieved validation test_correct
    for int_dim in intrinsic_array:
        print('###############################')
        print('Testing intrinsic dimension: {}'.format(int_dim))

        root = config.tasks[task_name]['root']
        stats = config.tasks[task_name]['stats']
        batch_size = config.tasks[task_name]['batch_size']

        train_loader, test_loader = get_loaders(root, task_name,
                                                True,
                                                batch_size,
                                                stats)

        # Get model and wrap it in fastfood
        if model_type == "cnn":
            model = RegularCNNModel().cuda()
        if model_type == "fcn":
            model = FCNAsInPAper().cuda()

        if he_normal:
            print("Using Kaiming He normal init")
            model = _kaiming_normal(model)

        if wrap_type == "fastfood":
            model = WrapFastfood(model, intrinsic_dimension=int_dim,
                                 device=config.device)
        if wrap_type == "dense":
            model = WrapDense(model, intrinsic_dimension=int_dim,
                                 device=config.device)

        writer.add_text(model_type, wrap_type, 0)

        # Train model and record highest validation
        grad_total = parameter_count(model)
        print('Parameter count, Grad: {}'.format(grad_total))

        model, optimizer = use_model(model, config.device, config.lr)
        if model_type == "cnn":
            # Add learning rate decay
            scheduler = ReduceLROnPlateau(optimizer, 'max', patience=6)

        loss = torch.nn.NLLLoss()

        trainer = create_supervised_trainer(model, optimizer, loss, device="cuda")
        train_evaluator = create_supervised_evaluator(model,
                                                metrics={
                                                    'accuracy': Accuracy(),
                                                    'nll': Loss(loss)
                                                }, device="cuda")
        valid_evaluator = create_supervised_evaluator(model,
                                                      metrics={
                                                          'accuracy': Accuracy(),
                                                          'nll': Loss(loss)
                                                      }, device="cuda")

        @trainer.on(Events.EPOCH_COMPLETED)
        def log_training_results(trainer):
            train_evaluator.run(train_loader)
            metrics = train_evaluator.state.metrics
            writer.add_scalar("Accuracy/train/IntDim: {}".format(int_dim), metrics['accuracy'], trainer.state.epoch)
            writer.add_scalar("Loss/train/IntDim: {}".format(int_dim), metrics['nll'], trainer.state.epoch)

        @trainer.on(Events.EPOCH_COMPLETED)
        def log_training_results(trainer):
            valid_evaluator.run(test_loader)
            metrics = valid_evaluator.state.metrics
            writer.add_scalar("Accuracy/test/IntDim: {}".format(int_dim), metrics['accuracy'], trainer.state.epoch)
            writer.add_scalar("Loss/test/IntDim: {}".format(int_dim), metrics['nll'], trainer.state.epoch)

            if model_type == "cnn":
                scheduler.step(metrics['accuracy'])

        def score_function(engine):
            acc_test = valid_evaluator.state.metrics['accuracy']
            return acc_test

        early_stop_handler = EarlyStopping(patience=8, score_function=score_function, trainer=trainer)
        valid_evaluator.add_event_handler(Events.COMPLETED, early_stop_handler)

        trainer.run(train_loader, max_epochs=config.n_epochs)

        writer.add_scalar("IntDimVSAcc", early_stop_handler.best_score, int_dim)

if __name__ == '__main__':
    main(config)