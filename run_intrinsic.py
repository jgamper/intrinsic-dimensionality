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
from src.utils import get_exponential_range
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.metrics import Accuracy, Loss
from ignite.handlers import EarlyStopping
import wandb

def main(dataset_path, results_path):
    """

    :return:
    """
    dataset_name = os.path.basename(dataset_path)
    assert dataset_name in list(config.tasks.keys()), "Don't have this dataset {}, or remove '/'".format(dataset_name)

    # Get intrinsic dimension options
    intrinsic_array = get_exponential_range(config.exp_max, config.num_max)

    wandb.init(project="intrinsic-dimensionality")

    wandb.config.dataset_name = dataset_name

    # Iterate over intrinsic dimension size and
    # record highest achieved validation test_correct
    for int_dim in intrinsic_array:
        print('###############################')
        print('Testing intrinsic dimension: {}'.format(int_dim))

        stats = config.tasks[dataset_name]['stats']
        batch_size = config.batch_size
        seed = config.seed
        num_classes = config.tasks[dataset_name]['num_classes']

        train_loader, test_loader = get_loaders(dataset_path,
                                                batch_size=batch_size,
                                                stats=stats,
                                                seed=seed)


        # Get model and wrap it in fastfood
        model = get_resnet("resnet18", num_classes).cuda()

        model = WrapFastfood(model, intrinsic_dimension=int_dim,
                             device=config.device)

        grad_total = parameter_count(model)
        print('Parameter count, Grad: {}'.format(grad_total))

        model, optimizer = use_model(model, config.device, config.lr)
        loss = torch.nn.NLLLoss()

        trainer = create_supervised_trainer(model, optimizer, loss, device="cuda")

        pbar = ProgressBar()
        pbar.attach(trainer, output_transform=lambda x: {'loss': x})

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
        def log_training_results(engine):
            train_evaluator.run(train_loader)
            metrics = train_evaluator.state.metrics
            wandb.log({"epoch".format(int_dim): engine.state.epoch, "int-dim: {}, train-acc": metrics['accuracy']})
            wandb.log({"epoch".format(int_dim): engine.state.epoch, "int-dim: {}, train-loss": metrics['nll']})

        @trainer.on(Events.EPOCH_COMPLETED)
        def log_validation_results(engine):
            valid_evaluator.run(test_loader)
            metrics = valid_evaluator.state.metrics
            wandb.log({"epoch".format(int_dim): engine.state.epoch, "int-dim: {}, test-acc": metrics['accuracy']})
            wandb.log({"epoch".format(int_dim): engine.state.epoch, "int-dim: {}, test-loss": metrics['nll']})
            pbar.log_message(
                "Validation Results - Epoch: {}  Avg score: {:.4f} Avg Loss: {:.2f}"
                    .format(engine.state.epoch, metrics['accuracy'], metrics['nll']))

        def score_function(engine):
            acc_test = valid_evaluator.state.metrics['accuracy']
            return acc_test

        early_stop_handler = EarlyStopping(patience=8, score_function=score_function, trainer=trainer)
        valid_evaluator.add_event_handler(Events.COMPLETED, early_stop_handler)

        trainer.run(train_loader, max_epochs=config.n_epochs)

        wandb.log({"intrinsic-dim": int_dim, "Accuracy": early_stop_handler.best_score})

if __name__ == '__main__':
    arguments = docopt(__doc__)
    dataset_path = arguments['--root']
    results_path = arguments['--res_dir']
    main(dataset_path, results_path)