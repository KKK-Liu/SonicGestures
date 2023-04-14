import serial

ser = serial.Serial('COM3', 9600)

while True:
    if ser.in_waiting > 0:
        data = ser.readline().rstrip().decode()
        num = int(data)
        # print(data)
        print(num)
