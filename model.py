import torch
import torch.nn as nn
import torch.nn.functional as F

def get_model(args):
    return myModel0414(args)
    
class myModel0414(torch.nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        # floor(N + 2P - K) / S + 1)
        # self.net = nn.Sequential(
        #     # nn.Linear(args.T * 25, 512),
        #     nn.Linear(5 * 25, 512),
        #     nn.ReLU(),
        #     nn.Linear(512,512),
        #     nn.ReLU(),
        #     nn.Linear(512,512),
        #     nn.ReLU(),
        #     nn.Linear(512,512),
        #     nn.ReLU(),
        #     nn.Linear(512,5),
        # )
        self.l1 = nn.Linear(5*25, 512)
        self.l2 = nn.Linear(512, 512)
        self.l3 = nn.Linear(512, 512)
        self.l4 = nn.Linear(512, 512)
        self.l5 = nn.Linear(512, 5)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x1 = F.relu(self.l1(x))
        x2 = F.relu(self.l2(x1))
        x3 = F.relu(self.l3(x2+x1))
        x4 = F.relu(self.l4(x3+x2+x1))
        x5 = F.log_softmax(self.l5(x4+x3+x2+x1), dim=1)
        return x5
    
if __name__ == '__main__':
    model = myModel0414(None)
    
    import time
    s = time.time()
    input = torch.rand(5,125)
    for _ in range(10000):
        output = model(input)
    e = time.time()
    print(f'{e-s:.3f}')
    # input = torch.rand(1,125)
    # output = model(input)
    # print(output)
    
    