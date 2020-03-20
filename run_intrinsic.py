import os
import argparse
from distutils.util import strtobool
from src.config import config
os.environ["CUDA_VISIBLE_DEVICES"] = str(config.device_id)
import torch
from intrinsic.fastfood import WrapFastfood
from src.models import get_resnet, _kaiming_normal
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
    parser.add_argument("-t", "--task_name", help="Task name")
    parser.add_argument("--he_normal", type=strtobool, default=False, help="If he normal init")
    args = parser.parse_args()
    task_name = args.task_name
    he_normal = args.he_normal

    # Sort out paths and get tensorboard write
    config = add_paths_to_config(config)

    writer = get_writer(config.tensorboard_dir)

    # Get intrinsic dimension options
    intrinsic_array = get_exponential_range(config.exp_max, config.num_max)

    # Iterate over intrinsic dimension size and
    # record highest achieved validation test_correct
    for int_dim in intrinsic_array:
        print('###############################')
        print('Testing intrinsic dimension: {}'.format(int_dim))

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
        model = WrapFastfood(model, intrinsic_dimension=int_dim,
                             device=config.device)

        writer.add_text(task_name, "Kaiming he normal: {}".format(he_normal), 0)

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

        def score_function(engine):
            acc_test = valid_evaluator.state.metrics['accuracy']
            return acc_test

        early_stop_handler = EarlyStopping(patience=8, score_function=score_function, trainer=trainer)
        valid_evaluator.add_event_handler(Events.COMPLETED, early_stop_handler)

        trainer.run(train_loader, max_epochs=config.n_epochs)

        writer.add_scalar("IntDimVSAcc", early_stop_handler.best_score, int_dim)

if __name__ == '__main__':
    main(config)