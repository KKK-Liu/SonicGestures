import torch
import torch.nn as nn

def get_model(args):
    return myModel0414(args)
    
class myModel0414(torch.nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        # floor(N + 2P - K) / S + 1)
        self.net = nn.Sequential(
            # nn.Linear(args.T * 25, 512),
            nn.Linear(10 * 25, 512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512,5),
        )
        
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.net(x)
    
if __name__ == '__main__':
    model = myModel0414(None)
    
    import time
    s = time.time()
    for _ in range(100):
        input = torch.rand(1,250)
        output = model(input)
    e = time.time()
    print(f'{e-s:.3f}')
    
    