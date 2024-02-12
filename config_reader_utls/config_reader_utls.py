import yaml


def read_file(file_path: str):
    with open(file_path, "r") as yaml_file:
        data = yaml.load(yaml_file, Loader=yaml.FullLoader)
        print("Read successful")
    return data
