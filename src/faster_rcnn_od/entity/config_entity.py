from dataclasses import dataclass
from pathlib import Path
from collections import namedtuple


@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_URL: str
    local_data_file: Path
    unzip_dir: Path
    
    
@dataclass(frozen=True)
class PrepareBaseModelConfig:
    root_dir: Path
    base_model_path: Path
    params_classes: int
    
    
DatatransformationConfig = namedtuple("data_transformation", [
  'root_dir',
  'local_data_file',
  'transformed_data_file'
])


trainConfig = namedtuple("train_faster_rcnn", [
'root_dir','model_loader_path',
 'train_loader_path','valid_loader_path',
 'outputs', 'params_epoches', 'params_lr',
 'params_momentum', 'params_weight_decay', 'params_batch_size'
])