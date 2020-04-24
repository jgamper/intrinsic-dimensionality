device = "cuda"
n_epochs = 100
exp_max = 5
num_max = 5
lr = 0.001
device_id = 1
batch_size = 32
seed = 32

tasks = {
        'breast': {'stats': {'mean': (0.6685, 0.4844, 0.6638), 'std': (0.2205, 0.2397, 0.1844)},
                     'num_classes': 2},
        'colon': {'stats': {'mean': (0.7348, 0.5770, 0.7010), 'std': (0.2241, 0.2842, 0.2276)},
                     'num_classes': 9},
        'lymphoma': {'stats': {'mean': (0.4125, 0.3381, 0.4149), 'std': (0.2068, 0.1937, 0.1913)},
                     'num_classes': 3},
        'lung': {'stats': {'mean': (0.7680, 0.6133, 0.7297), 'std': (0.1867, 0.2556, 0.2011)},
                      'num_classes': 6},
        'ovary': {'root': '/media/mount2/AllTheTasks/Pathology/PatchClassification/ovarian/split',
                  'stats': {'mean': (0.7301, 0.5199, 0.6619), 'std': (0.1690, 0.2129, 0.1583)},
                  'num_classes': 5},
        'oral': {'stats': {'mean': (0.7960, 0.7192, 0.8201), 'std': (0.1547, 0.1710, 0.1152)},
                   'num_classes': 3},
        'meningioma': {'stats': {'mean': (0.5648, 0.3882, 0.5284), 'std': (0.2492, 0.2009, 0.2280)},
                  'num_classes': 4}
         }
