import torch
import torch.nn as nn
import torch.nn.functional as F
import serial
import numpy as np


def get_model(args):
    return myModel0414(args)
    
class myModel0414(torch.nn.Module):
    def __init__(self, args) -> None:
        super().__init__()
        
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
    
class intergratedModel:
    def __init__(self, args) -> None:
        
        self.ser = serial.Serial(args.port, args.baud) # 串口名称和波特率
        
        self.model = get_model(args).cuda()
        self.model.load_state_dict(torch.load(args.ckpt_load_path)['state_dict'])

        self.actions = ['up','down','left','right','empty']
        
        self.model.eval()
                
    def get_action(self):
        with torch.no_grad():
            try:
                input = self.get_input(self.ser)
                prediction = model(input)
                action = torch.argmax(prediction)
                print(f'Action:{self.actions[action]}')
                return action
            except KeyboardInterrupt:
                print('Finish')
        
    def get_input(self, ser:serial.Serial, T=5):
        pack = []
        ser.flush()
        ser.reset_input_buffer()
        ser.reset_output_buffer()
        
        while True:
            if ser.in_waiting > 0:  # 检查串口是否有数据
                data = ser.readline().decode('utf-8').rstrip() # 读取数据并转换为字符串
                data = data.split(' ')
                # print(data)
                # exit()
                if len(data) != 25:
                    continue
                try:
                    data = [int(item) for item in data]
                except:
                    continue
                data = np.array(data)
                pack.append(data)
                if len(pack) == T:
                    break
        try:
            pack = np.stack(pack)
            # print(pack)
        except BaseException as e:
            print(pack)
            return None
        return pack
    
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
    
    