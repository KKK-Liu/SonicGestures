import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import os
from torch.utils.data import Dataset
import torch

class myDataset(Dataset):
    def __init__(self, data_root, debug=False) -> None:
        super().__init__()
        self.class_names = ['up','down','left','right','empty']
        
        self.data = []
        self.label = []
        
        for i, class_name in enumerate(self.class_names):
            data_names = os.listdir(os.path.join(data_root, class_name))
            for data_name in data_names:
                data = np.load(os.path.join(data_root, class_name, data_name))
                if debug:
                    print(data.shape,data_name)
                self.data.append(data)
                self.label += [i] * len(data)
            
        self.data = np.concatenate(self.data, axis=0)
        # self.data = torch.from_numpy(self.data)
        self.data = torch.tensor(self.data, dtype=torch.float32)
        self.label = torch.tensor(self.label).long()
        self.len = len(self.data)
        
        assert len(self.data) == len(self.label)
        
    def __getitem__(self, index):
        return self.data[index], self.label[index]
    
    def __len__(self):
        return self.len
    
os.makedirs('./gifs', exist_ok=True)
train_dataset = myDataset('./data/train')
test_dataset = myDataset('./data/test')

labels = [
        'down',
        'empty',
        'left',
        'right',
        'up',
    ]

for batch_n, (train_data, train_label) in enumerate(train_dataset):
    
    # for data_n, image in enumerate(train_data):
    data = train_data.numpy()

    print(data.shape)

    def animate(i):
        plt.clf() # 清除当前图形
        plt.title('Heatmap at t={}'.format(i))
        plt.imshow(data[i], interpolation='nearest',vmin=0,vmax=63)
        for (x, y), value in np.ndenumerate(data[i]):
            plt.text(y, x, round(value, 2), ha='center', va='center')
        plt.axis('off')

    fig = plt.figure(figsize=(4,6))
    plt.subplots_adjust(0,0,1,0.9,0,0)
    ani = animation.FuncAnimation(fig, animate, frames=5, interval=200, repeat=True)
    ani.save(f'./gifs/{batch_n}-{labels[int(train_label)]}.gif', writer='pillow')
    
    # exit()
    if batch_n > 100:
        break