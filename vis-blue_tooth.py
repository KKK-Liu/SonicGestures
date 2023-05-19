# import serial
# import time

# ser = serial.Serial('COM5', 115200) # 串口名称和波特率
# total_time = 0
# cnt = 0
# while True:
#     s = time.time()
#     if ser.in_waiting > 0:  # 检查串口是否有数据
#         data = ser.readline().decode('utf-8').rstrip() # 读取数据并转换为字符串
#         e = time.time()
#         print(f'{e-s:.4f} {data}')
        
        
import serial
import time
import numpy as np
from multiprocessing import Queue, Process
import numpy as np
import matplotlib.pyplot as plt

def listen(q:Queue):
    try:
        ser = serial.Serial('COM5', 115200) # 串口名称和波特率
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
    except:
        print("Error in Process Listen")
        q.put(-1)
            
def draw(q:Queue):
    mean = 0.0
    
    try:    
        while True:
            s = time.time()
            numbers = q.get()
            e = time.time()
            mean = 0.95*mean + 0.05*(e-s)
            print(numbers)
            print(f"{mean:.4f}")
            
    except:
        print("Error in Process Draw")
    

def main():
    q = Queue()
    p1 = Process(target=listen, args=(q,))
    p2 = Process(target=draw, args=(q,))
    p1.start()
    p2.start()

if __name__ == '__main__':
    main()
        

