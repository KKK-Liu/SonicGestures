import serial
import time

ser = serial.Serial('COM5', 9600) # 串口名称和波特率
total_time = 0
cnt = 0
while True:
    s = time.time()
    if ser.in_waiting > 0:  # 检查串口是否有数据
        data = ser.readline().decode('utf-8').rstrip() # 读取数据并转换为字符串
        e = time.time()
        print(f'{e-s:.4f} {data}')
        

