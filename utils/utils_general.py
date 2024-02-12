import os
import sys
import yaml
import torch
import logging
import numpy as np


def prep_folder(path: str, to_file: bool = False, absolute: bool = False):
    path_ = ''
    path_to_list_of_strings = path.split('/')
    if len(path_to_list_of_strings) < 2:
        sys.exit('Illegal path to file.')
    for i in range(len(path_to_list_of_strings) - 1):
        path_ += path_to_list_of_strings[i] + '/'
    if not to_file:
        path_ += path_to_list_of_strings[-1] + '/'
    if not absolute:
        path_ = os.getcwd() + '/' + path_
    os.makedirs(path_, exist_ok=True)
    return

def get_device(cuda_dev_id):
    """
    returns the device to use for computation, cuda if possible and the allocated request is correct, cpu otherwise
    :param cuda_dev_id: id of the desired cuda device, it could be a string or an int
    :return: the device to use for computation
    """
    if not torch.cuda.is_available():
        return torch.device('cpu')
    elif cuda_dev_id is None or int(cuda_dev_id) >= torch.cuda.device_count():
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        try:
            return torch.device('cuda:' + str(cuda_dev_id) if torch.cuda.is_available() else 'cpu')
        except ValueError:
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def logging_info(args, file_path):
    prep_folder(path=file_path, to_file=True)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(message)s",
                        handlers=[logging.FileHandler(file_path), logging.StreamHandler()]
                        )
    logger = logging.getLogger()
    logging.info('Args: %s', args)

    return logger

def from_numpy_to_dataloader(X, y, batch_size=100, shuffle=False):
    from torch.utils.data import TensorDataset, DataLoader

    tensor_x = torch.Tensor(X)  # transform to torch tensor
    tensor_y = torch.Tensor(y)

    dataset = TensorDataset(tensor_x, tensor_y)  # create your datset
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)  # create your dataloader

    return dataloader

def load_cfg_from_cfg_file(file: str):
    cfg = {}
    assert os.path.isfile(file) and (file.endswith('.yaml') or file.endswith('.yml')), '{} is not a valid file'.format(file)

    with open(file, 'r') as f:
        cfg_from_file = yaml.safe_load(f)

    for key in cfg_from_file:
        for k, v in cfg_from_file[key].items():
            cfg[k] = v
    cfg = CfgNode(cfg)
    return cfg


def from_datetime_to_string(datetime):
    elements_to_substitute = [':', ' ', '.', '-']
    new_el = ['_', '_', '_', '_']

    res = str(datetime)

    for i in range(len(elements_to_substitute)):
        res = res.replace(elements_to_substitute[i], new_el[i])

    return res


class CfgNode(dict):
    """
    CfgNode represents an internal node in the configuration tree. It's a simple
    dict-like container that allows for attribute-based access to keys.
    """

    def __init__(self, init_dict: dict = None):
        # Recursively convert nested dictionaries in init_dict into CfgNodes
        init_dict = {} if init_dict is None else init_dict
        super(CfgNode, self).__init__(init_dict)

    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        self[name] = value

    def __str__(self):
        def _indent(s_, num_spaces):
            string_ = s_.split("\n")
            if len(string_) == 1:
                return s_
            first = string_.pop(0)
            string_ = [(num_spaces * " ") + line for line in string_]
            string_ = "\n".join(string_)
            string_ = first + "\n" + string_
            return string_

        r = ""
        s = []
        for k, v in sorted(self.items()):
            separator = "\n" if isinstance(v, CfgNode) else " "
            attr_str = "{}:{}{}".format(str(k), separator, str(v))
            attr_str = _indent(attr_str, 2)
            s.append(attr_str)
        r += "\n".join(s)
        return r

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, super(CfgNode, self).__repr__())
