"""Usage:
          run_intrinsic.py [--root=<dataset_path>] [--gpu=<id>]

@ Jevgenij Gamper 2020
Sets up dvc with symlinks if necessary

Options:
  -h --help              Show help.
  --version              Show version.
  --root=<dataset_path>  Path to the dataset
  --gpu=<id>             GPU list. [default: 0]
"""
import os
from docopt import docopt
from src.config import config

def main(dataset_path):
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
            wandb.log({"epoch": engine.state.epoch, "int-dim: {}; train-acc".format(int_dim): metrics['accuracy']})
            wandb.log({"epoch": engine.state.epoch, "int-dim: {}; train-loss".format(int_dim): metrics['nll']})

        @trainer.on(Events.EPOCH_COMPLETED)
        def log_validation_results(engine):
            valid_evaluator.run(test_loader)
            metrics = valid_evaluator.state.metrics
            wandb.log({"epoch": engine.state.epoch, "int-dim: {}; test-acc".format(int_dim): metrics['accuracy']})
            wandb.log({"epoch": engine.state.epoch, "int-dim: {}; test-loss".format(int_dim): metrics['nll']})
            pbar.log_message(
                "Validation Results - Epoch: {}  Avg score: {:.4f} Avg Loss: {:.2f}"
                    .format(engine.state.epoch, metrics['accuracy'], metrics['nll']))

        def score_function(engine):
            acc_test = valid_evaluator.state.metrics['accuracy']
            return acc_test

        early_stop_handler = EarlyStopping(patience=8, score_function=score_function, trainer=trainer)
        valid_evaluator.add_event_handler(Events.COMPLETED, early_stop_handler)

        trainer.run(train_loader, max_epochs=config.n_epochs)

        wandb.log({"Final Accuracy": early_stop_handler.best_score}, step=int_dim)

if __name__ == '__main__':
    arguments = docopt(__doc__)
    dataset_path = arguments['--root']
    gpu = arguments['--gpu']

    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
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

    main(dataset_path)