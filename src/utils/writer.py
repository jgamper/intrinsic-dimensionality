from torch.utils.tensorboard import SummaryWriter

def get_writer(log_dir):
    """
    Returns summary writer and creates model graph image
    :param model:
    :param loader:
    :param log_dir:
    :return:
    """
    # Tensorboard
    writer = SummaryWriter(log_dir=log_dir)
    return writer
