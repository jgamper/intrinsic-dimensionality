import shutil
import datetime
import os


def make_dir(path_to_directory):
    """
    Makes a directory on a given path if one does not exist
    :param path_to_directory: path to directory
    :return:
    """
    if not os.path.exists(path_to_directory):
        os.makedirs(path_to_directory)


def create_results_dir(path_to_results_directory):
    """
    Copies all .py files into the results directory, just in case.
    Also allows to see what parameters in the config file were used.
    :param path_to_results_directory: path to results directory
    :return:
    """
    # Check if directory exists
    make_dir(path_to_results_directory)
    # Get time stamp
    today = datetime.datetime.today()
    time_stamp = today.strftime('%H-%M-%S-%Y-%m-%d').replace('-', '_')
    result_dir = os.path.join(path_to_results_directory, time_stamp)

    # Get path to src
    src = os.path.abspath(os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        os.pardir)
    )
    # # Copy all .py files
    shutil.copytree(
        src, result_dir, ignore=shutil.ignore_patterns(
            '*.pickle', '*.md', '*.ipynb', '*notebook*'
        )
    )

    out = os.path.join(result_dir, 'experiment_results')

    save_models_directory = os.path.join(out, 'model_snapshots')
    make_dir(save_models_directory)

    tensorboard_dir = os.path.join(out, 'tensorboard_dir')
    make_dir(tensorboard_dir)

    save_pred = os.path.join(out, 'predictions')
    make_dir(save_pred)

    return save_models_directory, tensorboard_dir, save_pred

def add_paths_to_config(config):
    """
    Add relevant paths to the config
    :param config:
    :return:
    """
    results_directory = config.results_directory


    # Get paths for saving models, and logging
    if config.checkpoint_timestamp:
        experiment_path = os.path.join(results_directory, config.checkpoint_timestamp, 'experiment_results')
        assert os.path.exists(experiment_path), 'The checkpointed experiment does not exist!'
        save_models_directory = os.path.join(experiment_path, 'model_snapshots')
        tensorboard_dir = os.path.join(experiment_path, 'tensorboard_dir')
        save_pred = os.path.join(experiment_path, 'predictions')
    else:
        save_models_directory, tensorboard_dir, save_pred = create_results_dir(results_directory)

    config.save_models_directory = save_models_directory
    config.tensorboard_dir = tensorboard_dir
    config.save_predictions_path = save_pred
    return config