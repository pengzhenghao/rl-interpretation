from toolbox.train import train, get_train_parser
from toolbox.utils import initialize_ray, get_local_dir, get_num_gpus, \
    get_num_cpus

ir = initialize_ray
