import os
import argparse
from src.config import config
from src.utils import add_paths_to_config, get_writer
from src.utils import train, test, use_model, parameter_count

def main(config):
    """

    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-id", "--device", help="GPU device id")
    parser.add_argument("-t", "--task_name", help="Task name")
    args = parser.parse_args()
    device_id = args.device
    task_name = args.task_name
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)

    # Now import pytorch
    from intrinsic.fastfood import WrapFastfood
    from src import (get_resnet, get_loaders, get_exponential_range)

    # Sort out paths and get tensorboard write
    config = add_paths_to_config(config)

    writer = get_writer(config.tensorboard_dir)

    # Get intrinsic dimension options
    intrinsic_array = get_exponential_range(config.exp_max, config.num_max)

    # Iterate over intrinsic dimension size and
    # record highest achieved validation score
    for int_dim in intrinsic_array:

        root = config.tasks[task_name]['root']
        means = config.tasks[task_name]['means']
        batch_size = config.tasks[task_name]['batch_size']
        num_classes = config.tasks[task_name]['num_classes']

        train_loader, test_loader = get_loaders(config.root, config.task_name,
                                                True,
                                                config.batch_size,
                                                config.means[config.task_name])


        # Get model and wrap it in fastfood
        model = get_resnet("resnet18", num_classes, pretrained=False).cuda()
        model = WrapFastfood(model, intrinsic_dimension=int_dim,
                             device=device_id)

        # Train model and record highest validation
        highest = 0
        grad_total, total = parameter_count(model)

        model, optimizer = use_model(model, config.device, config.lr)

        for epoch in range(1, config.n_epochs + 1):

            train_correct, train_loss = train(model, train_loader, optimizer,
                  config.epoch,
                  config.batch_log_interval,
                  config.device)

            test_correct, test_loss = test(model, test_loader, config.device)

            if test_correct > highest:
                highest = test_correct

            writer.add_scalar('Int dim {}/training/loss', train_loss, epoch)
            writer.add_scalar('Int dim {}/training/acc', train_correct, epoch)
            writer.add_scalar('Int dim {}/test/loss', test_loss, epoch)
            writer.add_scalar('Int dim {}/test/acc', test_correct, epoch)

        writer.add_hparams({'Intrinsic dim': int_dim,
                            'Total param': total,
                            'Grad param': grad_total},
                      {'Validation accuracy': highest})

if __name__ == '__main__':
    main(config)