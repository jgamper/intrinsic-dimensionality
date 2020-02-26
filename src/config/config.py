device = "cuda"
n_epochs = 100
exp_max = 5
num_max = 5
lr = 0.0003
batch_log_interval = 600
device_id = 0
patience = 3


tasks = {'CIFAR100': {'root': '/home/jevjev/Dropbox/Projects/datasets',
                    'stats': {'mean': (0.4914, 0.4822, 0.4465), 'std': (0.2023, 0.1994, 0.2010)},
                    'batch_size': 32,
                    'num_classes': 100},
        'CIFAR10': {'root': '/home/jevjev/Dropbox/Projects/datasets',
                    'stats': {'mean': (0.4914, 0.4822, 0.4465), 'std': (0.2023, 0.1994, 0.2010)},
                    'batch_size': 32,
                    'num_classes': 10}}

results_directory = "/home/jevjev/Int-dim-exp/"
