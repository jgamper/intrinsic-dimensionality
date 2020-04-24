"""Usage:
          run_intrinsic.py [--root=<dataset_path>]  [--res_dir=<results_path>]

@ Jevgenij Gamper 2020
Sets up dvc with symlinks if necessary

Options:
  -h --help              Show help.
  --version              Show version.
  --root=<dataset_path>  Path to the dataset
  --medical=<link>       Flag if training on medical data
  --res_dir=<results_path>  Directory to store results
"""
import os
from docopt import docopt
from src.config import config
os.environ["CUDA_VISIBLE_DEVICES"] = str(config.device_id)
import torch
from intrinsic.fastfood import WrapFastfood
from src.models import get_resnet
from src.data import get_loaders
from src.utils import use_model, parameter_count
from src.utils import get_writer, get_exponential_range
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss
from ignite.handlers import EarlyStopping

def main(dataset_path, results_path):
    """

    :return:
    """
    dataset_name = os.path.basename(dataset_path)
    writer = get_writer(results_path)

    # Get intrinsic dimension options
    intrinsic_array = get_exponential_range(config.exp_max, config.num_max)

    # Iterate over intrinsic dimension size and
    # record highest achieved validation test_correct
    for int_dim in intrinsic_array:
        print('###############################')
        print('Testing intrinsic dimension: {}'.format(int_dim))

        root = config.tasks[dataset_name]['root']
        stats = config.tasks[dataset_name]['stats']
        batch_size = config.tasks[dataset_name]['batch_size']
        num_classes = config.tasks[dataset_name]['num_classes']

        train_loader, test_loader = get_loaders(root,
                                                batch_size,
                                                stats)


        # Get model and wrap it in fastfood
        model = get_resnet("resnet18", num_classes).cuda()

        model = WrapFastfood(model, intrinsic_dimension=int_dim,
                             device=config.device)

        writer.add_text(dataset_name, "Kaiming he normal: {}".format(False), 0)

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
    arguments = docopt(__doc__)
    print(arguments)
    dataset_path = arguments['<dataset_path>']
    results_path = arguments['<results_path>']
    main(dataset_path, results_path)