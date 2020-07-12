import torch as t 
from torch.utils import data 
import os 
import h5py

class MCData(data.Dataset):
    def __init__(self,root,train=True):
        self.root = root
        datas = [os.path.join(root,data) for data in os.listdir(root)]
        data_num = len(datas)
        if train:
            self.train_dataset = datas[0:int(0.7*data_num)]
        else:
            self.train_dataset = datas[int(0.7*data_num):]
        
    def __getitem__(self,index):
        output = []
        data_path_object = self.train_dataset[index]
        with h5py.File(data_path_object,'r') as f:
            img_dataset = f['imgs_data']
            img_data_b = img_dataset[()]
            ref_dataset = f['ref_data']
            ref_data_b = ref_dataset[()]
        img_data_b = t.from_numpy(img_data_b)
        ref_data_b = t.from_numpy(ref_data_b)
        output.append(img_data_b)
        output.append(ref_data_b)
        return output
    
    def __len__(self):
        return len(self.train_dataset)