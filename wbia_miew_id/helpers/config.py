import yaml
from dataclasses import dataclass, field
from typing import List, Dict, Tuple

from dataclasses import asdict

def dataclass_to_dict(dataclass_instance):
    return asdict(dataclass_instance)

class DictableClass:
    def __iter__(self):
        yield from dataclass_to_dict(self).items()

@dataclass
class Train(DictableClass):
    anno_path: str
    n_filter_min: int = 4
    n_subsample_max: int = None

@dataclass
class Val(DictableClass):
    anno_path: str
    n_filter_min: int = 2
    n_subsample_max: int = 10

@dataclass
class Test(DictableClass):
    anno_path: str
    n_filter_min: int = 2
    n_subsample_max: int = 10
    checkpoint_path: str = None
    eval_groups: List = field(default_factory=list)

@dataclass
class PreprocessImages(DictableClass):
    apply: bool = False
    preprocessed_dir: str = None
    force_apply: bool = False

@dataclass
class Data(DictableClass):
    images_dir: str
    train: Train
    val: Val
    preprocess_images: PreprocessImages
    image_size: Tuple[int, int]
    test: Test = None
    viewpoint_list: List = None
    name_keys: List = field(default_factory=lambda: ['name'])
    crop_bbox: bool = False
    use_full_image_path: bool = False

@dataclass
class Engine(DictableClass):
    train_batch_size: int
    valid_batch_size: int
    epochs: int
    seed: int
    device: str
    use_wandb: bool
    num_workers: int = 0
    loss_module: str = 'softmax'
    use_swa: bool = False

@dataclass
class SWAParams(DictableClass):
    swa_lr: float = 0.00014
    swa_start: int = 21

@dataclass
class SchedulerParams(DictableClass):
    lr_start: float
    lr_max: float
    lr_min: float
    lr_ramp_ep: int
    lr_sus_ep: int
    lr_decay: float

@dataclass
class ModelParams(DictableClass):
    model_name: str
    use_fc: bool
    fc_dim: int
    dropout: float
    loss_module: str
    s: float
    margin: float
    ls_eps: float
    theta_zero: float
    pretrained: bool
    n_classes: int
    k: int = 2
    
@dataclass
class TestParams():
    batch_size: int = 4
    fliplr: bool = False
    fliplr_view: List = field(default_factory=list)

@dataclass
class Config(DictableClass):
    exp_name: str
    project_name: str
    checkpoint_dir: str
    comment: str
    data: Data
    engine: Engine
    scheduler_params: SchedulerParams
    model_params: ModelParams
    test: TestParams
    swa_params: SWAParams



def load_yaml(file_path: str) -> Dict:
    print(f"Loading config from path: {file_path}")
    with open(file_path, 'r') as file:
        config_dict = yaml.safe_load(file)

    return config_dict

def convert_config_dict(input_dict):
    
    input_dict['data']['train'] = {
        'anno_path': input_dict['data'].pop('train_anno_path'),
        'n_filter_min': input_dict['data'].pop('train_n_filter_min'),
        'n_subsample_max': input_dict['data'].pop('train_n_subsample_max')
    }
    
    input_dict['data']['val'] = {
        'anno_path': input_dict['data'].pop('val_anno_path'),
        'n_filter_min': input_dict['data'].pop('val_n_filter_min'),
        'n_subsample_max': input_dict['data'].pop('val_n_subsample_max')
    }

    input_dict['data']['test'] = {
        'anno_path': input_dict['data']['val']['anno_path'],
        'n_filter_min': input_dict['data']['val']['anno_path'],
        'n_subsample_max': input_dict['data']['val']['anno_path'],
        'checkpoint_path': ''
    }
    

    return input_dict


def get_config(file_path: str) -> Config:

    config_dict = load_yaml(file_path)

    if not config_dict['data'].get('train', False):
        print("Attempting to convert config dict to compatible format...")
        config_dict = convert_config_dict(config_dict)

    if not config_dict.get('swa_params', False):
        config_dict['swa_params'] = dict(SWAParams())

    if not config_dict['data'].get('preprocess_images', False) or isinstance(config_dict['data']['preprocess_images'], bool):
        config_dict['data']['preprocess_images'] = dict(PreprocessImages())

    config_dict['data'] = Data(**config_dict['data'])
    config_dict['data'].train = Train(**config_dict['data'].train)
    config_dict['data'].val = Val(**config_dict['data'].val)
    config_dict['data'].test = Test(**config_dict['data'].test)
    config_dict['data'].preprocess_images = PreprocessImages(**config_dict['data'].preprocess_images)
    config_dict['engine'] = Engine(**config_dict['engine'])
    config_dict['swa_params'] = SWAParams(**config_dict['swa_params'])
    config_dict['scheduler_params'] = SchedulerParams(**config_dict['scheduler_params'])
    config_dict['model_params'] = ModelParams(**config_dict['model_params'])
    config_dict['test'] = TestParams(**config_dict['test'])
    

    config = Config(**config_dict)
    return config

def write_config(config: Config, file_path: str):
    config_dict = dict(config)
    with open(file_path, 'w') as file:
        yaml.dump(config_dict, file)

def yaml_to_formatted_string(file_path):
    """
    Convert a YAML file to a formatted string.
    
    Args:
    file_path (str): The path to the YAML file.
    
    Returns:
    str: A formatted string representation of the YAML data.
    """
    try:
        # Read the YAML file
        with open(file_path, 'r') as file:
            data = yaml.safe_load(file)
        # Convert the dictionary to a formatted string
        formatted_str = yaml.dump(data, sort_keys=False, default_flow_style=False)
        return formatted_str
    except FileNotFoundError:
        return f"Error: File not found - {file_path}"
    except yaml.YAMLError as e:
        return f"Error parsing YAML: {e}"

def formatted_string_to_yaml(formatted_str, output_path):
    """
    Convert a formatted string to a YAML string and write it to a file.
    
    Args:
    formatted_str (str): A formatted string representation of data.
    output_path (str): The path where the YAML file will be saved.
    
    Returns:
    str: A message indicating success or failure.
    """
    try:
        # Load the formatted string into a Python dictionary
        data = yaml.safe_load(formatted_str)
        # Convert the dictionary back to a YAML string
        yaml_str = yaml.dump(data, sort_keys=False, default_flow_style=False)
        # Write the YAML string to the specified output file
        with open(output_path, 'w') as file:
            file.write(yaml_str)
        return f"YAML successfully written to {output_path}"
    except yaml.YAMLError as e:
        return f"Error parsing formatted string: {e}"
    except IOError as e:
        return f"Error writing to file: {e}"

