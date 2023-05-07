from faster_rcnn_od.config import ConfigurationManager
from faster_rcnn_od.components import training
from faster_rcnn_od import logger

class trainValPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        train_model_config = config.train_config()
        prepare_base_model = training(config=train_model_config)
        prepare_base_model.train_start()