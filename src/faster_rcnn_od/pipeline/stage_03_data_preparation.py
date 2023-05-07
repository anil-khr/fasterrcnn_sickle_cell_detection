from faster_rcnn_od.config import ConfigurationManager
from faster_rcnn_od.components import DataTransform_load
from faster_rcnn_od import logger

class DataPrepPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        prepare_data_config = config.get_data_transform_config()
        prepare_base = DataTransform_load(config=prepare_data_config)
        prepare_base.get_cust_data()