import torch
import time 
import os
from tqdm.auto import tqdm
from faster_rcnn_od.utils.comman import  Averager, SaveBestModel, save_loss_plot, save_model
from faster_rcnn_od.entity import trainConfig


class training:
    
    global train_itr
    global train_loss_list
    DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    def __init__(self, config: trainConfig):
        
        self.config = config
        print(self.config)
        
    def base_model(self):
        return torch.load(self.config.model_loader_path)
    
    def train_start(self):
        
        
        model = self.base_model()
        model = model.to(self.DEVICE)
        # get the model parameters
        params = [p for p in model.parameters() if p.requires_grad]
        # define the optimizer
        optimizer = torch.optim.SGD(params, lr=self.config.params_lr, momentum=self.config.params_momentum, weight_decay=self.config.params_weight_decay)
        # initialize the Averager class
        train_loss_hist = Averager()
        val_loss_hist = Averager()
        self.train_itr = 1
        self.val_itr = 1
        # train and validation loss lists to store loss values of all...
        # ... iterations till ena and plot graphs for all iterations
        # name to save the trained model with
        MODEL_NAME = 'model'
        # whether to show transformed images from data loader or not
        save_best_model = SaveBestModel()
        # start the training epochs
        NUM_EPOCHS = 1
        for epoch in range(NUM_EPOCHS):
            print(f"\nEPOCH {epoch+1} of {NUM_EPOCHS}")
            # reset the training and validation loss histories for the current epoch
            train_loss_hist.reset()
            val_loss_hist.reset()
            # start timer and carry out training and validation
            start = time.time()
            train_loss, loss_train = self.train(self.train_data_loader('train_loder.pt'), model, optimizer)
            val_loss, loss_val = self.validate(self.train_data_loader('val_loader.pt'), model)
            train_loss_hist.send(loss_train)
            train_loss_hist.send(loss_val)
            print(f"Epoch #{epoch+1} train loss: {train_loss_hist.value:.3f}")   
            print(f"Epoch #{epoch+1} validation loss: {val_loss_hist.value:.3f}")   
            end = time.time()
            print(f"Took {((end - start) / 60):.3f} minutes for epoch {epoch}")
            # save the best model till now if we have the least loss in the...
            # ... current epoch
            save_best_model(
                val_loss_hist.value, epoch, model, optimizer
            )
            # save the current epoch model
            save_model(epoch, model, optimizer)
            # save loss plot
            OUT_DIR =self.config.outputs
            save_loss_plot(OUT_DIR, train_loss, val_loss)
            
            # sleep for 5 seconds after each epoch
            time.sleep(5)
        
    def train_data_loader(self, file_name):  
        path = self.config.train_loader_path
        path_tr = os.path.join(path, file_name)
        data = torch.load(path_tr)
        for i in data:
            imgs = []
            targets = []
            for d in i:
                imgs.append(d[0])
                targ = {}
                targ['boxes'] = d[1]['boxes']#.to(device)
                targ['labels'] = d[1]['label']#.to(device)
                targets.append(targ)
        return (targets, imgs)
        
    def train(self,train_data_loader, model, optimizer):
        
        print('Training')
        global train_itr
        global train_loss_list
        train_loss_list = []
        
        # initialize tqdm progress bar
        prog_bar = tqdm(train_data_loader, total=len(train_data_loader))
        targets, images = train_data_loader
        for i, data in enumerate(prog_bar):
            
            optimizer.zero_grad()
            loss_dict = model(images, targets)
            print(loss_dict)
            losses = sum(loss for loss in loss_dict[0].values())
            loss_value = losses.item()
            train_loss_list.append(loss_value)
            train_loss_hist = Averager()
            train_loss_hist.send(loss_value)
            losses.backward()
            optimizer.step()
            self.train_itr += 1
            
        
            # update the loss value beside the progress bar for each iteration
            prog_bar.set_description(desc=f"Loss: {loss_value:.4f}")
        return train_loss_list, loss_value
    
    
    def validate(self, valid_data_loader, model ):
        print('Validating')
        global val_itr
        global val_loss_list
        val_loss_list = []
        
        # initialize tqdm progress bar
        prog_bar = tqdm(valid_data_loader, total=len(valid_data_loader))
        
        for i, data in enumerate(prog_bar):
            targets, images  = valid_data_loader
            
            
            with torch.no_grad():
                
                loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict[0].values())
            loss_value = losses.item()
            val_loss_list.append(loss_value)
            
           # val_loss_hist.send(loss_value)
            self.val_itr += 1
            # update the loss value beside the progress bar for each iteration
            prog_bar.set_description(desc=f"Loss: {loss_value:.4f}")
        return val_loss_list, loss_value
     
        