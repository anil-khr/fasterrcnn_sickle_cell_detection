import torchvision
import torch
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from faster_rcnn_od.entity import PrepareBaseModelConfig




class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config

    
    def get_base_model(self):
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained = True)
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.config.params_classes)
       
        
        self.save_model(self.config.base_model_path, self.model)
    

    def save_model(self,path, model):
        model = torch.jit.script(model)
        model.save(path)    