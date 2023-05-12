# fasterrcnn_sickle_cell_detection

## Workflows

1. Update config.yaml
2. Update secrets.yaml [Optional]
3. Update params.yaml
4. Update the entity
5. Update the configuration manager in src config
6. Update the components
7. Update the pipeline 
8. Update the main.py


##

How to run?

STEPS:

Clone the repository

https://github.com/anil-khr/fasterrcnn_sickle_cell_detection.git

STEP 01- Create a conda environment after opening the repository

conda create -n fasterrcnn python=3.8 -y
conda activate fasterrcnn


STEP 02- install the requirements

pip install -r requirements.txt

# Finally run the following command

python app.py





The params.yaml file contains the following information:

The batch size to use for training. If you carry out training on your own system, then you may increase or decrease the size according to your available GPU memory.

The dimensions that we want the images to resize to, that is RESIZE_TO.

Number of epochs to train for. The models included in the zip file for download are trained for 10 epochs as well.

Number of workers or sub-processes to use for data loading. This helps a lot when we have a large dataset or reading images form disk, or even doing a lot of image augmentations as well.

The computation device to use for training. For training, you will need a GPU. A CPU is just too slow for Faster RCNN training and object detection training in general as well.

Then we have the classes and the number of classes. Note that we have a __background__ class at the beginning. This is required while fine-tuning PyTorch object detection models. The class at index 0 is always the __background__ class.
