from faster_rcnn_od.constants import *
from faster_rcnn_od.utils.comman import read_yaml, create_directories
from faster_rcnn_od.entity import (DataIngestionConfig, PrepareBaseModelConfig, DatatransformationConfig, trainConfig)
from pathlib import Path
import os




class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])

    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_URL=config.source_URL,
            local_data_file=config.local_data_file,
            unzip_dir=config.unzip_dir 
        )

        return data_ingestion_config
    
    


    def get_prepare_base_model_config(self) -> PrepareBaseModelConfig:
        config = self.config.prepare_base_model
        
        create_directories([config.root_dir])

        prepare_base_model_config = PrepareBaseModelConfig(
            root_dir=Path(config.root_dir),
            base_model_path=Path(config.base_model_path),
            params_classes=self.params.CLASSES
            
        )
    
        return prepare_base_model_config
        
        


    
    def get_data_transform_config(self) -> DatatransformationConfig:
        config = self.config.data_transformation

        create_directories([config.root_dir])

        data_transformation_config = DatatransformationConfig(
            root_dir=config.root_dir,
            local_data_file=config.local_data_file,
            transformed_data_file=config.transformed_data_file 
        )

        return data_transformation_config
    
    
    def train_config(self) -> trainConfig:
        config = self.config.train_faster_rcnn

        create_directories([config.root_dir])

        train_config = trainConfig (
            root_dir=config.root_dir,
            model_loader_path=config.model_loader_path,
            train_loader_path=config.train_loader_path,
            valid_loader_path= config.valid_loader_path,
            outputs= config.outputs,
            params_epoches=self.params.EPOCHS,
            params_lr=self.params.LEARNING_RATE,
            params_momentum=self.params.MOMENTUM,
            params_weight_decay = self.params.WEIGHT_DECAY,
            params_batch_size = self.params.BATCH_SIZE
        )

        return train_config
    
    
    