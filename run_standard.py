import os
import argparse
from distutils.util import strtobool
from src.config import config
os.environ["CUDA_VISIBLE_DEVICES"] = str(config.device_id)
from intrinsic.fastfood import WrapFastfood
import torch
from src.models import get_resnet, _kaiming_normal
from src.data import get_loaders
from src.utils import use_model, parameter_count
from src.utils import get_writer, add_paths_to_config, get_exponential_range
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss, RunningAverage
from ignite.handlers import EarlyStopping
from ignite.contrib.handlers import ProgressBar


def main(config):
    """

    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--task_name", help="Task name")
    parser.add_argument("--int_dim", default=None)
    parser.add_argument("--he_normal", type=strtobool, default=False, help="If he normal init")
    args = parser.parse_args()
    task_name = args.task_name
    he_normal = args.he_normal
    int_dim = args.int_dim
    # Sort out paths and get tensorboard write
    config = add_paths_to_config(config)

    writer = get_writer(config.tensorboard_dir)

    # Get intrinsic dimension options
    intrinsic_array = get_exponential_range(config.exp_max, config.num_max)

    # Iterate over intrinsic dimension size and
    # record highest achieved validation test_correct
    root = config.tasks[task_name]['root']
    stats = config.tasks[task_name]['stats']
    batch_size = config.tasks[task_name]['batch_size']
    num_classes = config.tasks[task_name]['num_classes']

    train_loader, test_loader = get_loaders(root, task_name,
                                            True,
                                            batch_size,
                                            stats)


    # Get model and wrap it in fastfood
    model = get_resnet("resnet18", num_classes).cuda()
    if he_normal:
        print("Using Kaiming He normal init")
        model = _kaiming_normal(model)
    writer.add_text(task_name, "Kaiming he normal: {}".format(he_normal), 0)
    if int_dim:
        int_dim = int(int_dim)
        print("Intrinsic dimension: {}".format(int_dim))
        model = WrapFastfood(model, intrinsic_dimension=int_dim,
                             device=config.device)

    grad_total = parameter_count(model)
    print('Parameter count, Grad: {}'.format(grad_total))

    model, optimizer = use_model(model, config.device, config.lr)
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

    RunningAverage(output_transform=lambda x: x).attach(trainer, "loss")

    pbar = ProgressBar(persist=True)
    pbar.attach(trainer, metric_names="all")

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_results(trainer):
        iter = trainer.state.iteration
        if iter % 100 == 0:
            train_evaluator.run(train_loader)
            metrics = train_evaluator.state.metrics
            writer.add_scalar("Accuracy/train/", metrics['accuracy'], trainer.state.epoch)
            writer.add_scalar("Loss/train/", metrics['nll'], trainer.state.epoch)

            pbar.log_message(
                "Training Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}".format(
                    trainer.state.epoch, metrics['accuracy'], metrics['nll']
                )
            )

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_test_results(trainer):
        iter = trainer.state.iteration
        if iter % 100 == 0:

            valid_evaluator.run(test_loader)
            metrics = valid_evaluator.state.metrics
            writer.add_scalar("Accuracy/test/", metrics['accuracy'], trainer.state.epoch)
            writer.add_scalar("Loss/test/", metrics['nll'], trainer.state.epoch)

            pbar.log_message(
                "Validation Results - Epoch: {}  Avg accuracy: {:.2f} Avg loss: {:.2f}".format(
                    trainer.state.epoch, metrics['accuracy'], metrics['nll']
                )
            )

            pbar.n = pbar.last_print_n = 0

    def score_function(engine):
        acc_test = valid_evaluator.state.metrics['accuracy']
        return acc_test

    early_stop_handler = EarlyStopping(patience=8, score_function=score_function, trainer=trainer)
    valid_evaluator.add_event_handler(Events.COMPLETED, early_stop_handler)

    trainer.run(train_loader, max_epochs=config.n_epochs)

if __name__ == '__main__':
    main(config)