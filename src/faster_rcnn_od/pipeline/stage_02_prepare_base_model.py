from faster_rcnn_od.config import ConfigurationManager
from faster_rcnn_od.components import PrepareBaseModel
from faster_rcnn_od import logger

class BaseModelPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        prepare_base_model_config = config.get_prepare_base_model_config()
        prepare_base_model = PrepareBaseModel(config=prepare_base_model_config)
        prepare_base_model.get_base_model()