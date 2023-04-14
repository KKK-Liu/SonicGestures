import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os


class myDataset(Dataset):
    def __init__(self, data_root) -> None:
        super().__init__()
        self.class_names = ['up','down','left','right','empty']
        
        self.data = []
        self.label = []
        
        for i, class_name in enumerate(self.class_names):
            for data_name in os.listdir(os.path.join(data_root, class_name)):
                data = np.load(os.path.join(data_root, class_name, data_name))
                self.data.append(data)
                self.label += [i] * len(data)
            
        self.data = np.concatenate(self.data, axis=0)
        self.data = torch.from_numpy(self.data)
        self.label = torch.tensor(self.label).long()
        self.len = len(self.data)
        
        assert len(self.data) == len(self.label)
        
    def __getitem__(self, index):
        return self.data[index], self.label[index]
    
    def __len__(self):
        return self.len
    
def get_dataloader(args):
    data_loader_args = {
        'batch_size':args.batch_size,
        'num_workers':args.num_workers,
        'shuffle':True,
        'pin_memory':True,
        'prefetch_factor':4,
        'persistent_workers':True
    }
    
    train_dataset = myDataset(os.path.join(args.data_root, 'train'))
    test_dataset = myDataset(os.path.join(args.data_root, 'test'))
    
    train_dataloder = DataLoader(
        train_dataset,
        **data_loader_args
    )
    test_dataloder = DataLoader(
        test_dataset,
        **data_loader_args
    )
    
    return train_dataloder, test_dataloder