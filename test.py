import torch
from model import get_model
from arguments import get_args
from tqdm import tqdm
import serial
import numpy as np


def main():
    assert torch.cuda.is_available(), "how could you do not use cuda???"
    '''
        Initialization!
    '''
    
    args = get_args()
    ser = serial.Serial(args.port, args.baud) # 串口名称和波特率
    
    model = get_model(args).cuda()
    model.load_state_dict(torch.load(args.ckpt_load_path)['state_dict'])

    
    actions = ['up','down','left','right','empty']
    
    '''
        Validation!
    '''
    model.eval()
    with torch.no_grad():
        while True:
            try:
                input = get_input(ser)
                prediction = model(input)
                action = torch.argmax(prediction)
                print(f'Action:{actions[action]}')
            except KeyboardInterrupt:
                print('Finish')
                

def get_input(ser:serial.Serial, T=5):
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
    main()