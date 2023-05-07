from faster_rcnn_od.entity import DatatransformationConfig
from torch.utils.data import DataLoader
import pandas as pd
from faster_rcnn_od.utils.comman import custom_collate, CustDat
import torch
import os



class DataTransform_load:
    def __init__(self, config: DatatransformationConfig ):
        self.config = config

    
    def transform_file(self):
        
        if  os.path.exists(self.config.local_data_file):
            self.data = pd.read_csv(self.config.local_data_file)
            self.df = self.data[['filename','xmin', 'ymin', 'xmax','ymax', 'class']]
            self.df['class'] = self.df['class'].map({'Sickle': 1, 'Normal': 2,'Target': 3, 'Other': 4, 'Crystal': 5 })
            self.unique_imgs = self.df['filename'].unique()
            from sklearn.model_selection import train_test_split
            self.train_inds, self.val_inds = train_test_split(range(self.unique_imgs.shape[0]), test_size = 0.2)
    
            #self.save_file(self.config.transformed_data_file, self.df)
            return self.df, self.unique_imgs, self.train_inds, self.val_inds
            #return self.unique_imgs
            
    def get_cust_data(self):
        df, img, train_idx, val_idx = self.transform_file()
        self.train_dl = DataLoader(CustDat(df, img, train_idx),
                              batch_size=1,
                              shuffle=True,
                              collate_fn = custom_collate,
                              pin_memory = False)
        
        self.val_dl = DataLoader(CustDat(df, img, val_idx),
                              batch_size=1,
                              shuffle=True,
                              collate_fn = custom_collate,
                              pin_memory = False)
        self.save_file(self.config.transformed_data_file, self.train_dl, 'train_loder')
        self.save_file(self.config.transformed_data_file, self.val_dl, 'val_loader')

            
    

    def save_file(self, path, train_dl, file_name):
        print(file_name)
        path_trn = os.path.join(path, file_name)
        torch.save(train_dl, path_trn+".pt")
        
