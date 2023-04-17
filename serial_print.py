import serial

ser = serial.Serial('COM3', 9600) # 串口名称和波特率
while True:
    if ser.in_waiting > 0:  # 检查串口是否有数据
        data = ser.readline().decode('utf-8').rstrip() # 读取数据并转换为字符串
        print(data) # 输出数据
