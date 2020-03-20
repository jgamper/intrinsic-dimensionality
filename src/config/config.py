device = "cuda"
n_epochs = 100
exp_max = 5
num_max = 5
lr = 0.001
batch_log_interval = 600
device_id = 1

tasks = {'MNIST': {'root': '/home/jevjev/Dropbox/Projects/datasets',
                    'stats': {'mean': (0.1307), 'std': (0.3081)},
                    'batch_size': 32,
                    'num_classes': 10},
        'CIFAR100': {'root': '/home/jevjev/Dropbox/Projects/datasets',
                    'stats': {'mean': (0.4914, 0.4822, 0.4465), 'std': (0.2023, 0.1994, 0.2010)},
                    'batch_size': 32,
                    'num_classes': 100},
        'CIFAR10': {'root': '/home/jevjev/Dropbox/Projects/datasets',
                    'stats': {'mean': (0.4914, 0.4822, 0.4465), 'std': (0.2023, 0.1994, 0.2010)},
                    'batch_size': 32,
                    'num_classes': 10},
        'BREAST': {'root': '/media/mount2/AllTheTasks/Pathology/PatchClassification/breast/split',
                     'stats': {'mean': (0.6685, 0.4844, 0.6638), 'std': (0.2205, 0.2397, 0.1844)},
                     'batch_size': 32,
                     'num_classes': 2},
        'COLON': {'root': '/media/mount2/AllTheTasks/Pathology/PatchClassification/colon/split',
                             'stats': {'mean': (0.7348, 0.5770, 0.7010), 'std': (0.2241, 0.2842, 0.2276)},
                             'batch_size': 32,
                             'num_classes': 9},
        'LYMPHOMA': {'root': '/media/mount2/AllTheTasks/Pathology/PatchClassification/lymphoma/split',
                             'stats': {'mean': (0.4125, 0.3381, 0.4149), 'std': (0.2068, 0.1937, 0.1913)},
                             'batch_size': 32,
                             'num_classes': 3}
         }

results_directory = "/home/jevjev/Int-dim-exp/"
