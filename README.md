
# WILDBOOK IA - MiewID Plugin

A plugin for matching and interpreting embeddings for wildlife identification.


## Setup

``` 
pip install -r requirements.txt
pip install -e .
```

Optionally, these environment variables must be set to enable Weights and Biases logging
capability:
```
WANDB_API_KEY={your_wanb_api_key}
WANDB_MODE={'online'/'offline'}
```

## Multispecies-V2 Model

Model specs and dataset overview can be found at the [model card page for the Multispecies-v2 model](https://huggingface.co/conservationxlabs/miewid-msv2)

### Pretrained Model Embeddings Extraction

```
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
from transformers import AutoModel

model_tag = f"conservationxlabs/miewid-msv2"
model = AutoModel.from_pretrained(model_tag, trust_remote_code=True)

def generate_random_image(height=440, width=440, channels=3):
    random_image = np.random.randint(0, 256, (height, width, channels), dtype=np.uint8)
    return Image.fromarray(random_image)

random_image = generate_random_image()

preprocess = transforms.Compose([
    transforms.Resize((440, 440)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

input_tensor = preprocess(random_image)
input_batch = input_tensor.unsqueeze(0) 

with torch.no_grad():
    output = model(input_batch)

print(output)
print(output.shape)

```

### Pretrained Model Evaluation
```
import torch
from wbia_miew_id.evaluate import Evaluator
from transformers import AutoModel

evaluator = Evaluator(
    device=torch.device('cuda'),
    seed=0,
    anno_path='beluga_example_miewid/benchmark_splits/test.csv',
    name_keys=['name'],
    viewpoint_list=None,
    use_full_image_path=True,
    images_dir=None,
    image_size=(440, 440),
    crop_bbox=True,
    valid_batch_size=12,
    num_workers=8,
    eval_groups=[['species', 'viewpoint']],
    fliplr=False,
    fliplr_view=[],
    n_filter_min=2,
    n_subsample_max=10,
    model_params=None,
    checkpoint_path=None,
    model=model,
    visualize=False,
    visualization_output_dir='beluga_example_visualizations'
)
```

## Example Usage

### Example dataset download

```
cd wbia_miew_id
python examples/download_example.py
```

### Training

```
python train.py --config=examples/beluga_example_miewid/benchmark_model/miew_id.msv2_all.yaml
```

### Evaluation

```
python evaluate.py --config=examples/beluga_example_miewid/benchmark_model/miew_id.msv2_all.yaml
```

Optional `--visualize` flag can be used to produce top 5 match results for each individual in the test set, along with gradcam visualizations.

### Data Splitting, Training, and Evaluation Using Python Bindings

Demo notebooks are avaliable at [examples directory](https://github.com/WildMeOrg/wbia-plugin-miew-id/tree/main/wbia_miew_id/examples)

## Data files

### Example dataset

The data is expected to be in the CSV or COCO JSON Format. 

[Recommended] The CSV beluga data can be downlaoded from [here](https://cthulhu.dyn.wildme.io/public/datasets/beluga_example_miewid.tar.gz).

The COCO beluga data can be downloaded from [here](https://cthulhu.dyn.wildme.io/public/datasets/beluga-model-data.zip).

### Expected CSV data format

- `theta`: Bounding box rotation in radians
- `viewpoint`: Viewpoint of the individual facing the camera. Used for calculating per-viewpoint stats or separating individuals based on viewpoint
- `name`: Individual ID
- `file_name`: File name
- `viewpoint`: Species name. Used for calculating per-species stats
- `file_path`: Full path to images
- `x, y, w, h`: Bounding box coordinates

|theta         |viewpoint                       |name |file_name|species|file_path|x    |y                                                                                                                   |w   |h  |
|--------------|--------------------------------|-----|---------|-------|---------|-----|--------------------------------------------------------------------------------------------------------------------|----|---|
|0             |up                              |1030 |000000006040.jpg|beluga_whale|/datasets/beluga-440/000000006040.jpg|0    |0                                                                                                                   |162 |440|
|0             |up                              |1030 |000000006043.jpg|beluga_whale|/datasets/beluga-440/000000006043.jpg|0    |0                                                                                                                   |154 |440|
|0             |up                              |508  |000000006044.jpg|beluga_whale|/datasets/beluga-440/000000006044.jpg|0    |0                                                                                                                   |166 |440|


## Configuration file

A config file path can be set by:
`python train.py --config {path_to_config}`

- `exp_name`: Name of the experiment
- `project_name`: Name of the project
- `checkpoint_dir`: Directory for storing training checkpoints
- `comment`: Comment text for the experiment
- `viewpoint_list`: List of viewpoint values to keep for all subsets.
- `data`: Subfield for data-related settings
  - `images_dir`: Directory containing the all of the dataset images
  - `use_full_image_path`: Overrides the images_dir for path construction and instead uses an absolute path that should be defined in the `file_path` file path under the `images` entries for each entry in the COCO JSON. In such a case, `images_dir` can be set to `null`
  - `crop_bbox`: Whether to use the bounding box metadata to crop the images. The crops will also be adjusted for rotation if the `theta` field is present for the annotations
  - `preprocess_images` pre-applies cropping and resizing and caches the images for training
  - `train`: Data parameters regarding the train set used in train.py
    - `anno_path`: Path to the JSON file containing the annotations
    - `n_filter_min`: Minimum number of samples per name (individual) to keep that individual in the set. Names under the threshold will be discarded
    - `n_subsample_max`: Maximum number of samples per name to keep for the training set. Annotations for names over the threshold will be randomly subsampled once at the start of training
  - `val`: Data parameters regarding the validation set used in train.py
    - `anno_path`
    - `n_filter_min`
    - `n_subsample_max`
  - `test`: Data parameters regarding the test set used in test.py
    - `anno_path`
    - `n_filter_min`
    - `n_subsample_max`
    - `checkpoint_path`: Path to model checkpoint to test
    - `eval_groups`: Attributes for which to group the testing sets. For example, the value of `['viewpoint']` will create subsets of the test set for each unique value of the viewpoint and run one-vs-all evaluation for each subset separately. The value can be a list - `[['species', 'viewpoint']]` will run evaluation separately for each species+viewpoint combination. `['species', 'viewpoint']` will run grouped eval for each species, and then for each viewpoint. The corresponding fields to be grouped should be present under `annotation` entries in the COCO file. Can be left as `null` to do eval for the full test set.
  - `name_keys`: List of keys used for defining a unique name (individual). Fields from multiple keys will be combined to form the final representation of a name. A common use-case is `name_keys: ['name', 'viewpoint']` for treating each name + viewpoint combination as a unique individual
  - `image_size`:
    - Image height to resize to
    - Image width to resize to
- `engine`: Subfields for engine-related settings
  - `num_workers`: Number of workers for data loading (default: 0)
  - `train_batch_size`: Batch size for training
  - `valid_batch_size`: Batch size for validation
  - `epochs`: Number of training epochs
  - `seed`: Random seed for reproducibility
  - `device`: Device to be used for training
  - `use_wandb`: Whether to use Weights and Biases for logging
  - `use_swa`: Whether to use SWA during training
- `scheduler_params`: Subfields for  learning rate scheduler parameters
  - `lr_start`: Initial learning rate
  - `lr_max`: Maximum learning rate
  - `lr_min`: Minimum learning rate
  - `lr_ramp_ep`: Number of epochs to ramp up the learning rate
  - `lr_sus_ep`: Number of epochs to sustain the maximum learning rate
  - `lr_decay`: Rate of learning rate decay per epoch
- `model_params`: Dictionary containing model-related settings
  - `model_name`: Name of the model backbone architecture
  - `use_fc`: Whether to use a fully connected layer after backbone extraction
  - `fc_dim`: Dimension of the fully connected layer
  - `dropout`: Dropout rate
  - `loss_module`: Loss function module
  - `s`: Scaling factor for the loss function
  - `margin`: Margin for the loss function
  - `pretrained`: Whether to use a pretrained model backbone
  - `n_classes`: Number of classes in the training dataset, used for loading checkpoint
- `swa_params`: Subfields for SWA training
  - `swa_lr`: SWA learning rate
  - `swa_start`: Epoch number to begin SWA training
- `test`: Subfields for plugin-related settings
  - `fliplr`: Whether to perform horizontal flipping during testing
  - `fliplr_view`: List of viewpoints to apply horizontal flipping
  - `batch_size`: Batch size for plugin inference
  
