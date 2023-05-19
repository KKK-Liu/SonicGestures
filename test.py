# import torch
from model import get_model
from arguments import get_args
from tqdm import tqdm
import serial
import numpy as np
from multiprocessing import Queue, Process
import torch
import pyautogui

'''
    This is the real online situation test.
'''
PORT = 'COM5'
BAUDRATE = 115200
T = 5

action = ['up','down','left','right','empty']

def listen(q:Queue):
    try:
        ser = serial.Serial(PORT, BAUDRATE) # 串口名称和波特率
        while True:
            if ser.in_waiting > 0:  # 检查串口是否有数据
                data = ser.readline().decode('utf-8').rstrip().lstrip() # 读取数据并转换为字符串
                numbers = data.split(' ')
                if len(numbers) != 24:
                    print('Error number. continue.')
                    continue
                numbers = numbers[::-1]
                numbers = np.array(list(map(int, numbers))).reshape((4,6)).T
                q.put(numbers)
                print('Listen Put')
    except:
        print("Error in Process Listen")
        q.put(-1)
    
    
class myQueue:
    def __init__(self, T) -> None:
        self.T = T
        self.q = []
        
    def put(self, x):
        if len(self.q) < self.T:
            self.q.append(x)
            print('1')
        else:
            print(2)
            self.q.pop(0)
            self.q.append(x)
    
    def can_get(self):
        print(self.T == len(self.q))
        return self.T == len(self.q)

    def get(self):
        # print(np.array(self.q).shape)
        return np.array(self.q).astype(np.float32)
        
        
        
def calculate(q_listen:Queue, q_control:Queue):
    myq = myQueue(T)
    myModel = get_model()
    
    while True:
        numbers = q_listen.get()
        print('Calculate Get')
        myq.put(numbers)
        if myq.can_get():
            input = myq.get()
            input = torch.tensor(input).unsqueeze(0)
            print(input.shape)
            output = myModel(input)
            print(output.shape)
            output = output.squeeze(0)
            action = int(torch.argmax(output))
            q_control.put(action)
            print('Control put')
    
def control(q:Queue):
    
    '''
        We assume that most of the times, the hand gesture is empty
        and the motion of up, down, left, down takes no less than 3 time slice
        
        We use finite state machine to control the state transfer
        
    '''
    state = 0
    # root = r"./imgs/FSM.png"
    current_action = 0
    
    while True:
        value = q.get()
        print('Control Get')
        print(f'Current State:{state} action:{action[value]} ',end='')
        if state == 0:
            if action[value] != 'empty':
                state = 1
                current_action = action[value]
                
        elif state == 1:
            if action[value] == current_action:
                state = 2
            else:
                state = 0
            
        elif state == 2:
            if action[value] == current_action:
                state = 3
            else:
                state = 0
                
        elif state == 3:
            print(f"Action:{current_action}")
            # pyautogui.press([current_action])
            state = 4
            
        elif state == 4:
            if action[value] == 'empty':
                state = 5
            else:
                state = 4
                
        elif state == 5:
            if action[value] == 'empty':
                state = 6
            else:
                state = 4
        
        elif state == 6:
            if action[value] == 'empty':
                state = 0
            else:
                state = 4
        print(f'Next State:{state}')

    
def main():
    q_listen_calculate = Queue()
    q_calculate_control = Queue()
    
    p_listen = Process(target=listen, args = (q_listen_calculate, ))
    p_calculate = Process(target=calculate, args = (q_listen_calculate,q_calculate_control ))
    p_control = Process(target=control, args = (q_calculate_control, ))
    
    p_listen.start()
    p_calculate.start()
    p_control.start()
    
    

if __name__ == '__main__':
    main()
    
# def main():
#     '''
#         Initialization!
#     '''
    
#     args = get_args()
#     ser = serial.Serial(args.port, args.baud) # 串口名称和波特率
    
#     model = get_model(args)
#     # model.load_state_dict(torch.load(args.ckpt_load_path)['state_dict'])

    
#     actions = ['up','down','left','right','empty']
    
#     '''
#         Test!
#     '''
#     model.eval()
#     with torch.no_grad():
#         while True:
#             try:
#                 input = get_input(ser)
#                 prediction = model(input)
#                 action = torch.argmax(prediction)
#                 print(f'Action:{actions[action]}')
#             except KeyboardInterrupt:
#                 print('Finish')
#                 break
                

# def get_input(ser:serial.Serial, T=5):
#     pack = []
#     ser.flush()
#     ser.reset_input_buffer()
#     ser.reset_output_buffer()
    
#     while True:
#         if ser.in_waiting > 0:  # 检查串口是否有数据
#             data = ser.readline().decode('utf-8').rstrip() # 读取数据并转换为字符串
#             data = data.split(' ')
#             # print(data)
#             # exit()
#             if len(data) != 25:
#                 continue
#             try:
#                 data = [int(item) for item in data]
#             except:
#                 continue
#             data = np.array(data)
#             pack.append(data)
#             if len(pack) == T:
#                 break
#     try:
#         pack = np.stack(pack)
#         # print(pack)
#     except BaseException as e:
#         print(pack)
#         return None
#     return pack

