import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os


class myDataset(Dataset):
    def __init__(self, data_root, debug=False, isTrain=True) -> None:
        super().__init__()
        self.class_names = ['up','down','left','right','empty']
        self.isTrain = isTrain
        self.data = []
        self.label = []
        
        for i, class_name in enumerate(self.class_names):
            data_names = os.listdir(os.path.join(data_root, class_name))
            this_num = 0
            for data_name in data_names:
                data = np.load(os.path.join(data_root, class_name, data_name))
                if debug:
                    print(data.shape,data_name)
                self.data.append(data)
                self.label += [i] * len(data)
                this_num += len(data)
            print(f'data root:{data_root} {class_name}:{this_num}')
            
        self.data = np.concatenate(self.data, axis=0)
        # self.data = torch.from_numpy(self.data)
        self.data = torch.tensor(self.data, dtype=torch.float32)
        self.label = torch.tensor(self.label).long()
        self.len = len(self.data)
        
        assert len(self.data) == len(self.label)
        
    def __getitem__(self, index):
        if self.isTrain:
            return self.data[index] + torch.randn(self.data[0].shape)*0.5 + torch.randint(low = - int(torch.min(self.data[index])), high =max(0, 64-int(torch.max(self.data[index]))), size=(1,)),\
            self.label[index]
        else:
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
    
    train_dataset = myDataset(os.path.join(args.data_root, 'train'), isTrain=True)
    test_dataset = myDataset(os.path.join(args.data_root, 'test'), isTrain=False)
    
    train_dataloder = DataLoader(
        train_dataset,
        **data_loader_args
    )
    test_dataloder = DataLoader(
        test_dataset,
        **data_loader_args
    )
    
    return train_dataloder, test_dataloder

def dataloader_test():
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='./data')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=2)
    
    args = parser.parse_args()
    
    train_dataloader, test_dataloader = get_dataloader(args)
    
    print(len(train_dataloader))
    print(len(test_dataloader))
    
    for data, label in train_dataloader:
        print(data.shape)
        print(label.shape)
        print(data)
        print(label)
        break
    
    for data, label in test_dataloader:
        print(data.shape)
        print(label.shape)
        print(data)
        print(label)
        break
    
if __name__ == '__main__':
    # dataloader_test()
    arr = torch.randn((4,4))
    print(torch.min(arr))
    print(torch.max(arr))
    print(arr)