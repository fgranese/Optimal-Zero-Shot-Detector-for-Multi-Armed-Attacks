from pipeline_testing import test
from pipeline_training import train
from config_reader_utls import attrDict
from config_reader_utls import config_reader_utls


if __name__ == '__main__':

    file_path = 'config/config.yaml'
    global_dict = config_reader_utls.read_file(file_path=file_path)
    attr_dict = attrDict.AttrDict.from_nested_dicts(global_dict)
    print(attr_dict)
    if attr_dict.TRAIN.switch:
        for loss_name in ['CE', 'KL', 'Rao', 'g']:
            train(attr_dict, loss_name)
    test(attr_dict)

