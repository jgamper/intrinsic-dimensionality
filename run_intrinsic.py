import os
import argparse
from src.config import config
os.environ["CUDA_VISIBLE_DEVICES"] = str(config.device_id)
from intrinsic.fastfood import WrapFastfood
from src.models import get_resnet, get_resnet_mean_var
from src.data import get_loaders
from src.utils import train, test, use_model, parameter_count
from src.utils import get_writer, add_paths_to_config, get_exponential_range

def main(config):
    """

    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--task_name", help="Task name")
    args = parser.parse_args()
    task_name = args.task_name

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
        model = get_resnet_mean_var("resnet18", num_classes).cuda()
        model = WrapFastfood(model, intrinsic_dimension=int_dim,
                             device=config.device)

        # Train model and record highest validation
        highest = 0
        counter = 0
        early_stop = False
        grad_total = parameter_count(model)
        print('Parameter count, Grad: {}'.format(grad_total))

        model, optimizer = use_model(model, config.device, config.lr)

        for epoch in range(1, config.n_epochs + 1):
            if early_stop:
                print('Stopping')
                break

            train_correct, train_loss = train(model, train_loader, optimizer,
                  epoch,
                  config.batch_log_interval,
                  config.device)

            test_correct, test_loss = test(model, test_loader, config.device)

            if test_correct < highest:
                counter += 1
                if counter >= config.patience:
                    print('Early stopping should happen')
                    early_stop = True
            else:
                highest = test_correct

            writer.add_scalar('Int dim {}/training/loss', train_loss, epoch)
            writer.add_scalar('Int dim {}/training/acc', train_correct, epoch)
            writer.add_scalar('Int dim {}/test/loss', test_loss, epoch)
            writer.add_scalar('Int dim {}/test/acc', test_correct, epoch)

        writer.add_hparams({'Intrinsic dim': int_dim},
                      {'Validation accuracy': highest})

if __name__ == '__main__':
    main(config)