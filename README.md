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

### How to run?

### STEPS:

Clone the repository

https://github.com/anil-khr/fasterrcnn_sickle_cell_detection.git

### STEP 01- Create a conda environment after opening the repository

conda create -n fasterrcnn python=3.8 -y
conda activate fasterrcnn


### STEP 02- install the requirements

pip install -r requirements.txt

### Finally run the following command

python main.py





### The params.yaml file contains the following information:

The batch size to use for training. If you carry out training on your own system, then you may increase or decrease the size according to your available GPU memory.

The dimensions that we want the images to resize to, that is RESIZE_TO.

Number of epochs to train for. The models included in the zip file for download are trained for 10 epochs as well.

Number of workers or sub-processes to use for data loading. This helps a lot when we have a large dataset or reading images form disk, or even doing a lot of image augmentations as well.

The computation device to use for training. For training, you will need a GPU. A CPU is just too slow for Faster RCNN training and object detection training in general as well.

Then we have the classes and the number of classes. Note that we have a __background__ class at the beginning. This is required while fine-tuning PyTorch object detection models. The class at index 0 is always the __background__ class.




### The utils/comman.py file contains the following helper functions and utility classes.



#### Averager class:
Use the Averager class to keep track of training and validation loss values and retrieve the average loss after each epoch. If you're new to this type of class.

#### SaveBestModel class: 
The SaveBestModel function is a simple way to save the best model after each epoch. Simply call the instance of this class and pass the current epoch's validation loss. If it's the best loss, a new best model will be saved to disk.

#### collate_fn:
The collate_fn() will help us take care of tensors of varying sizes while creating the training and validation data loaders.

#### read_yaml
This function reads a YAML file and returns the content as a ConfigBox class instance.

#### create_directories
This function creates directories at the given paths.

#### CustDat
This class is used as a PyTorch data loader

#### others
Then we have function to save the model after each epoch and after training ends.
And another function to save the training and validation loss graphs. This will be called after each epoch.

### Dataset for RCNN ResNet50 model
The RCNN ResNet50 FPN model requires a specific CSV data format for training. The CSV file should contain annotations for each image in the dataset, where each row corresponds to one object in an image. The CSV file should have the following format:

#### The first row should contain column headers: filename, width, height, class, xmin, ymin, xmax, ymax.

The filename column should contain the name of the image file, including the extension.

The width and height columns should contain the dimensions of the image in pixels.

The class column should contain the name of the object class.

The xmin, ymin, xmax, and ymax columns should contain the coordinates of the bounding box for the object. The coordinates should be in the format of (xmin, ymin, xmax, ymax) and should be normalized to the range of 0 to 1, where (0, 0) is the top left corner of the image and (1, 1) is the bottom right corner.


## The Faster RCNN Model with ResNet50 Backbone
The model preparation part is quite easy and straightforward. PyTorch already provides a pre-trained Faster RCNN ResNet50 FPN model. So, we just need to:

Load that model.
Get the number of input features.
And add a new head with the correct number of classes according to our dataset.

